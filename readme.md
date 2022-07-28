# Image Retoucher REST
_REST API powering the Alexa "Image Retoucher" skill._

## Supported
* Currently hosts a few static images to test with, enumerated via `/api/image/static`
* Supports commands enumerated via `/meta/help/commands`
* Append commands to the url of an image like so: `url/of/image/exposure/-50/tint/25` to reduce exposure by 50% and slide tint 25% up.
* Append `/comparison` to any valid render URL to get a side-by-side comparison with the original image

## Not yet supported
* `POST` to `/api/image` with image data to upload it to the backend DB. Returns a JSON object whose `url` field has the URL of the newly-available image
* `POST` to `/valid/render/url/save` to render that image and add it to the backend DB

# Support
Open an Issue on this repository for questions or bugs.