import logging
import os
import io
from typing import Callable, Tuple, Type
from time import perf_counter
from flask import Flask, Response, url_for, request, jsonify, send_file
import numpy as np
import cv2 as cv
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

def adjust_nop(img: np.ndarray, cmd, val):
    app.logger.debug(f'No-op executed for command \'{cmd}\' with parameter \'{val}\'')
    return img

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
    )
}

STATIC_IMAGES = {
    0 : 'static/belgium.jpg',
    1 : 'static/helens.jpg',
    2 : 'static/japan.jpg',
    3 : 'static/lake.jpg',
    4 : 'static/moon.jpg',
    5 : 'static/snowman.jpg',
}

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
    static_path = STATIC_IMAGES.get(id)
    img = Response(ERROR_PAGE, 404)
    if (static_path is None):
        # TODO try retrieving key from postgres
        pass
    else:
        app.logger.info(f'Loading static resource {static_path}...')
        img = cv.imread(static_path, cv.IMREAD_COLOR)
    return img

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
ERROR_PAGE = "<h1 style='color:red'>Internal Server Error</h1>"


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

@app.route("/api/image/<int:id>/")
def api_image_id_get(id:int):
    _img = get_image(id)
    if _img is None:
        return Response(ERROR_PAGE, 500)
    if isinstance(_img, Response):
        return _img

    _success, _buffer = cv.imencode('.png', _img)
    if not _success:
        return Response(ERROR_PAGE, 500)

    return send_file(
        io.BytesIO(_buffer),
        mimetype='image/png'
    )

@app.route("/api/image/<int:id>/<path:path>")
def api_image_id_get_path(id:int, path):
    _img = get_image(id)
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

    _success, _buffer = cv.imencode('.png', _img)
    if not _success:
        return Response(ERROR_PAGE, 500)

    return send_file(
        io.BytesIO(_buffer),
        mimetype='image/png'
    )

@app.route("/api/image/<int:id>/<path:path>/comparison")
def api_image_id_get_path_comparison(id:int, path):
    _img = get_image(id)
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
    _success, _buffer = cv.imencode('.png', _stack)
    if not _success:
        return Response(ERROR_PAGE, 500)

    return send_file(
        io.BytesIO(_buffer),
        mimetype='image/png'
    )

@app.route("/api/image/<int:id>/<path:path>/save", methods=['POST'])
def api_image_id_path_save(id:int, path):
    _img = get_image(id)
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

    _id = 0 # TODO upload image to postgres
    return jsonify({
        "url": url_for(api_image_id_get.__name__, id=_id)
    })
