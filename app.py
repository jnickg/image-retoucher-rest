import logging
import os
import io
from typing import Callable, Tuple, Type, List
from time import perf_counter
from flask import Flask, Response, url_for, request, jsonify, send_file
import numpy as np
import cv2 as cv
from scipy.interpolate import UnivariateSpline
import postgres

#
# Flask server setup
#

app = Flask(__name__)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

if __name__ != "__main__":
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

#
# Helpers & Constants
#
def spline_lut(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    normalized = np.array((data - data_min) / (data_max - data_min), dtype=np.float32)
    return normalized

def do_color_transfer(src_img : np.ndarray, tgt_img : np.ndarray):
    src_star = np.zeros_like(src_img, dtype=np.float32)
    for i in range(3):
        src_star[:,:,i] = src_img[:,:,i] - np.mean(src_img[:,:,i])
    sigma_s = [np.std(src_img[:,:,i]) for i in range(3)]
    sigma_t = [np.std(tgt_img[:,:,i]) for i in range(3)]
    src_prime = np.zeros_like(src_img, dtype=np.float32)
    for i in range(3):
        src_prime[:,:,i] = sigma_t[i] / sigma_s[i] * src_star[:,:,i]
    result = np.zeros_like(src_img, dtype=np.float32)
    for i in range(3):
        result[:,:,i] = src_prime[:,:,i] + np.mean(tgt_img[:,:,i])
    return result

def intparam2float(value:int) -> float:
    return float((value + 100) * 0.01 )

def intparam2tint(value:int) -> float:
    return float((value) * 0.01 * (360))

def adjust_exposure(img: np.ndarray, value:int):
    img = cv.convertScaleAbs(img, img, beta=value)
    return img

def adjust_contrast(img: np.ndarray, value:int):
    bright_factor = intparam2float(value)
    img = cv.convertScaleAbs(img, img, alpha=bright_factor)
    return img

def adjust_saturation(img: np.ndarray, value:int):
    sat_factor = intparam2float(value)
    hsv = cv.cvtColor(img.astype(dtype=np.float32), cv.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1] * sat_factor
    hsv[:,:,1] = np.clip(hsv[:,:,1], a_min=0.0, a_max=1.0)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR).astype(dtype=np.uint8)
    return img

def adjust_tint(img: np.ndarray, value:int):
    tint_slide = intparam2tint(value)
    hsv = cv.cvtColor(img.astype(dtype=np.float32), cv.COLOR_BGR2HSV)
    hsv[:,:,0] = np.mod(hsv[:,:,0] + tint_slide, 360.0)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR).astype(dtype=np.uint8)
    return img

def apply_clahe(img: np.ndarray):
    Lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    _l, _a, _b = cv.split(Lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    _l_equalized = clahe.apply(_l)
    lab_equalized = cv.merge((_l_equalized,_a,_b))
    img_equalized = cv.cvtColor(lab_equalized, cv.COLOR_LAB2BGR)
    return img_equalized

def apply_colortransfer(src: np.ndarray, tgt_id:int):
    tgt = get_image(tgt_id)

    src_lab = cv.cvtColor(normalize_data(src), cv.COLOR_BGR2LAB)
    tgt_lab = cv.cvtColor(normalize_data(tgt), cv.COLOR_BGR2LAB)
    result_lab = do_color_transfer(src_lab, tgt_lab)
    result = cv.cvtColor(result_lab, cv.COLOR_LAB2BGR)
    result = (result * 255.0).astype(dtype=np.uint8)
    return result

def apply_grayscale(src: np.ndarray):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    bgr_gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    return bgr_gray # Preserve 3-channel format so other operations behave as expected

def apply_sharpen(src: np.ndarray):
    src_f = src.astype(dtype=np.float32) / 255.0
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    result_f = cv.filter2D(src_f, -1, kernel)
    result_f = np.clip(result_f, a_min=0.0, a_max=1.0)
    return (result_f * 255.0).astype(dtype=np.uint8)

def apply_hdr(src: np.ndarray):
    return cv.detailEnhance(src, sigma_s=12, sigma_r=0.15)

def adjust_nop(img: np.ndarray, cmd, val):
    app.logger.debug(f'No-op executed for command \'{cmd}\' with parameter \'{val}\'')
    return img

def apply_summer_filter(img: np.ndarray):
    increase_lut = spline_lut([0, 64, 128, 256], [0, 80, 160, 256])
    decrease_lut = spline_lut([0, 64, 128, 256], [0, 50, 100, 256])
    _b, _g, _r  = cv.split(img)
    _r = cv.LUT(_r, increase_lut).astype(np.uint8)
    _b = cv.LUT(_b, decrease_lut).astype(np.uint8)
    sum = cv.merge((_b, _g, _r ))
    return sum

def apply_winter_filter(img: np.ndarray):
    increase_lut = spline_lut([0, 64, 128, 256], [0, 80, 160, 256])
    decrease_lut = spline_lut([0, 64, 128, 256], [0, 50, 100, 256])
    _b, _g, _r  = cv.split(img)
    _r = cv.LUT(_r, decrease_lut).astype(np.uint8)
    _b = cv.LUT(_b, increase_lut).astype(np.uint8)
    sum = cv.merge((_b, _g, _r ))
    return sum

class CommandDescriptor:
    def __init__(self,
                description : str = "N/A",
                func : Callable[[np.ndarray, any],np.ndarray] = None,
                param_range : Tuple = None,
                param_type : Type = None):
        self.description = description
        self.func = func
        self.param_range = param_range
        self.param_type = param_type

COMMAND = {
    'exposure': CommandDescriptor(
        description="Adjust the exposure (brightness) of the image",
        func = lambda i, v: adjust_exposure(i, int(v)),
        param_type = int,
        param_range = (-100,100)
    ),
    'contrast': CommandDescriptor(
        description="Adjust the contrast of the image",
        func = lambda i, v: adjust_contrast(i, int(v)),
        param_type = int,
        param_range = (-100, 100)
    ),
    'saturation': CommandDescriptor(
        description="Adjust the saturation of the image (S channel in HSV space)",
        func = lambda i, v: adjust_saturation(i, int(v)),
        param_type = int,
        param_range = (-100, 100)
    ),
    'tint': CommandDescriptor(
        description="Adjust the tint (hue) of the image (H channel in HSV space). Cyclic range (-100 == 0 == +100)",
        func = lambda i, v: adjust_tint(i, int(v)),
        param_type = int,
        param_range = (-100, 100)
    ),
    'clahe': CommandDescriptor(
        description="Apply contrast-limited adaptive histogram equalization to the image (applied to L channel in CIELAB space). Param can be any value (ignored).",
        func = lambda i, _: apply_clahe(i),
        param_type = type(None),
        param_range = (None, None)
    ),
    'colorxfer': CommandDescriptor(
        description="Apply color-transfer to the current image in CIELAB space, using the specified image ID",
        func = lambda i, tgt_id: apply_colortransfer(i, int(tgt_id)),
        param_type = int,
        param_range = (0, None)
    ),
    'gray': CommandDescriptor(
        description="Apply color2gray algorithm using canonical OpenCV algorithm.",
        func = lambda i, _: apply_grayscale(i),
        param_type = type(None),
        param_range = (None, None)
    ),
    'sharpen': CommandDescriptor(
        description="Apply basic Laplacian sharpen.",
        func = lambda i, _: apply_sharpen(i),
        param_type = type(None),
        param_range = (None, None)
    ),
    'hdr': CommandDescriptor(
        description="Apply a single-image HDR-ish algorithm to enhance detail.",
        func = lambda i, _: apply_hdr(i),
        param_type = type(None),
        param_range = (None, None)
    ),
    'summer': CommandDescriptor(
        description="Apply a summery color filter.",
        func = lambda i, _: apply_summer_filter(i),
        param_type = type(None),
        param_range = (None, None)
    ),
    'winter': CommandDescriptor(
        description="Apply a wintry color filter.",
        func = lambda i, _: apply_winter_filter(i),
        param_type = type(None),
        param_range = (None, None)
    )
}

STATIC_IMAGES = {
    0 : 'static/belgium.png',
    1 : 'static/car.png',
    2 : 'static/dragon.png',
    3 : 'static/helens.png',
    4 : 'static/japan.png',
    5 : 'static/lake.png',
    6 : 'static/moon.png',
    7 : 'static/temple.png',
    8 : 'static/trees.png',
    9 : 'static/wizard.png'
}

STATIC_COLLAGE = None

STATIC_IMAGEDATA = [None] * len(STATIC_IMAGES.keys())

def process_path(img: np.ndarray, path):
    i = np.array(path.split('/'))
    commands = i[0::2]
    params = i[1::2]
    cmd_pairs = list(zip(commands, params)) # listify so we can iterate twice
    app.logger.debug(f'Parsed commands: {cmd_pairs}')
    start = perf_counter()
    for (cmd, param) in cmd_pairs:
        command = COMMAND.get(cmd, CommandDescriptor(description="No-op", func = lambda i, v: adjust_nop(i, cmd, param)))
        cmdstart = perf_counter()
        img = command.func(img, param)
        cmdend = perf_counter()
        app.logger.debug(f'Command {cmd} processed in {cmdend-cmdstart:0.5f}s')
    end = perf_counter()
    app.logger.info(f'Total processing time: {end-start:0.5f}s')
    return img

def get_image(id:int) -> np.ndarray:
    global STATIC_IMAGES
    global STATIC_IMAGEDATA
    
    static_path = STATIC_IMAGES.get(id)
    img = Response(ERROR_PAGE, 404)
    if (static_path is None):
        # TODO try retrieving key from postgres
        pass
    else:
        if STATIC_IMAGEDATA[id] is not None:
            return STATIC_IMAGEDATA[id]
        app.logger.info(f'Loading static resource {static_path}...')
        img = cv.imread(static_path, cv.IMREAD_COLOR)
        STATIC_IMAGEDATA[id] = img
    return img

def encode_image(img: np.ndarray, download_name=None) -> any:
    _success, _buffer = cv.imencode('.png', img)
    if not _success:
        return Response(ERROR_PAGE, 500)

    return send_file(
        io.BytesIO(_buffer),
        mimetype='image/png',
        download_name=download_name
    )

def resize_image(img, size=(28,28)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv.resize(img, size, cv.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv.INTER_AREA if dif > (size[0]+size[1])//2 else cv.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv.resize(mask, size, interpolation)

def make_collage(images:List[np.ndarray]) -> np.ndarray:
    global STATIC_COLLAGE
    
    if STATIC_COLLAGE is not None:
        return STATIC_COLLAGE
    COLLAGE_SQUARE_SIDE = 400
    COLLAGE_SQUARE_DIMS = (COLLAGE_SQUARE_SIDE, COLLAGE_SQUARE_SIDE)
    COLLAGE_SQUARE_EMPTY = (COLLAGE_SQUARE_SIDE, COLLAGE_SQUARE_SIDE, 3)
    images = [resize_image(_img, COLLAGE_SQUARE_DIMS) for _img in images]
    added_blank = False
    if (len(images) % 2) != 0:
        images.append(np.zeros(COLLAGE_SQUARE_EMPTY))
        added_blank = True
    COLLAGE_RATIO = 3.0 / 5.0
    img_count = len(images)
    factors = []
    for fac_1 in range(1, img_count+1):
        if img_count % fac_1 == 0:
            fac_2 = img_count // fac_1
            factors.append((fac_1, fac_2))
    factor_ratios = [(f1/f2) for (f1, f2) in factors]
    ratio_dists = [abs(r - COLLAGE_RATIO) for r in factor_ratios]
    best_dimensions = factors[np.argmin(ratio_dists)]
    (h, w) = best_dimensions
    h_px = h * COLLAGE_SQUARE_SIDE
    w_px = w * COLLAGE_SQUARE_SIDE
    app.logger.debug(f'Creating collage with dimensions h={h_px} w={w_px}')
    collage = np.zeros((h_px, w_px, 3))
    for h_i in range(h):
        for w_i in range(w):
            idx = (h_i * w) + w_i
            y = h_i * COLLAGE_SQUARE_SIDE
            x = w_i * COLLAGE_SQUARE_SIDE
            collage[y:y+COLLAGE_SQUARE_SIDE,x:x+COLLAGE_SQUARE_SIDE,:] = images[idx]
            if not added_blank or idx != img_count - 1:
                collage = cv.putText(collage, f'{idx}', (x, y+50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(0,255,0), thickness=5)
    STATIC_COLLAGE = collage
    return collage


ROOT_PAGE = """
<h1 style='color:blue'>Image Retoucher REST API</h1>
<hr />
<p><i><a href="https://github.com/jnickg/image-retoucher-rest">Home page</a>.</i></p>
<p>REST API powering the Alexa "Image Retoucher" skill.</p>
<p><b>Helpful links:</b></p>
<ul>
    <li>Navigate to <a href="/api/image/static"><code>/api/image/static</code></a> for a list of static test images.</li>
    <li>Navigate to <a href="/meta/help/commands"><code>/meta/help/commands</code></a> for a list of available commands.</li>
</ul>

"""

ERROR_PAGE = """
<h1 style='color:red'>Internal Server Error</h1>
"""


#
# Auxiliary Routes
#

@app.route("/")
def _():
    return ROOT_PAGE

@app.route("/meta/help/commands")
def meta_help_commands():
    commands = []
    for key in COMMAND.keys():
        command_descriptor = COMMAND[key]
        new_command = {
            "name": key,
            "description": command_descriptor.description,
            "type": command_descriptor.param_type.__name__,
            "range": command_descriptor.param_range,
            "example_url": url_for(api_image_id_get_path.__name__, id=0, path=f'{key}/{command_descriptor.param_range[0]}')
        }
        commands.append(new_command)
    return jsonify({
        "commands":commands
    })

#
# API Routes
#

@app.route("/api/image", methods=['POST'])
def api_image_post():
    _data = np.fromstring(request.data, np.uint8)
    _img = cv.imdecode(_data, cv.IMREAD_COLOR)
    _id = 0 # TODO upload image to postgres
    return jsonify({
        "url": url_for(api_image_id_get.__name__, id=_id)
    })

@app.route("/api/image/static")
def api_image_static():
    ids = STATIC_IMAGES.keys()
    urls = [url_for(api_image_id_get.__name__, id=_id) for _id in ids]
    names = [STATIC_IMAGES[_id] for _id in ids]
    return jsonify({
        "count" : len(urls),
        "urls": urls,
        "names": names
    })

@app.route("/api/image/static/collage")
def api_image_static_collage():
    ids = STATIC_IMAGES.keys()
    images = [get_image(_id) for _id in ids]
    collage = make_collage(images)
    return encode_image(collage)

@app.route("/api/image/<int:id>/")
def api_image_id_get(id:int):
    _img = get_image(id).copy()
    if _img is None:
        return Response(ERROR_PAGE, 500)
    if isinstance(_img, Response):
        return _img

    return encode_image(_img)

@app.route("/api/image/<int:id>/<path:path>")
def api_image_id_get_path(id:int, path):
    _img = get_image(id).copy()
    if _img is None:
        return Response(ERROR_PAGE, 500)
    if isinstance(_img, Response):
        return _img

    try:
        app.logger.info(f'Processing path: {path}')
        _img = process_path(_img, path)
    except Exception as e:
        app.logger.error(f'Failed to execute commands in URL. Error: {e}')
        return Response(ERROR_PAGE, 500)

    return encode_image(_img, download_name=f"{id}_{str(path).replace('/','_')}.png")

@app.route("/api/image/<int:id>/<path:path>/comparison")
def api_image_id_get_path_comparison(id:int, path):
    _img = get_image(id).copy()
    if _img is None:
        return Response(ERROR_PAGE, 500)
    if isinstance(_img, Response):
        return _img

    _orig = _img.copy()
    try:
        app.logger.info(f'Processing path: {path}')
        _img = process_path(_img, path)
    except Exception as e:
        app.logger.error(f'Failed to execute commands in URL. Error: {e}')
        return Response(ERROR_PAGE, 500)

    _stack = np.hstack((_orig, _img))
    return encode_image(_stack, download_name=f"{id}_{str(path).replace('/','_')}_comparison.png")

@app.route("/api/image/<int:id>/<path:path>/save", methods=['POST'])
def api_image_id_path_save(id:int, path):
    _img = get_image(id)
    if _img is None:
        return Response(ERROR_PAGE, 500)
    if isinstance(_img, Response):
        return _img

    # Not implemented
    return Response(ERROR_PAGE, 501)
