# TIDL Python API Demo

This is my attempt at reimplementing the [BeagleBone AI TIDL demo][] using the
Python API for TIDL.  This is still a work in progress.  See
<https://www.glennklockwood.com/embedded/beaglebone-tidl.html> for more
detailed notes.

To get started, first download the model, parameters, and sample input by
running

    ./download_datasets.sh

Then run the webcam classifier service:

    sudo ./classify_webcam.py 2>&1 | grep -v "Corrupt JPEG data"

This will open an mjpg stream on http://localhost:5000/ which shows the webcam's
view with an overlay of whatever the top classification from ImageNet as long as
the confidence is above 25%.  In your terminal, you should see the frame rate
of classification printed out every ten seconds or so.

## Core code

- `classify_webcam.py` - my implementation of the BeagleBone AI classification
  demo using TIDL's Python API and Flask instead of the TIDL C++ API and
  mjpg-streamer.
- `imagenet.py` - The TI example of using the Python API to do classification on
  which the BeagleBone AI demo is modeled.  A modified version of
  <https://git.ti.com/cgit/tidl/tidl-api/tree/examples/pybind/imagenet.py>

## Helpful tools:

- `show_tidl_api.py` - Dumps the TIDL Python interfaces to stdout
- `stream_webcam.py` - Shows how to connect OpenCV's Python API to Flask and
  stream webcam video over HTTP.
- `download_datasets.sh` - Copy required example input and model+parameters into
  this directory.  They aren't stored in this git repository because they're
  big, binary, and don't change.
- `classify_url.sh` - Pass a URL to an image and this script will download and
  identify it using `imagenet.py`

[BeagleBone AI TIDL demo]: https://github.com/glennklockwood/beaglebone-ai/blob/main/classification/classification.cpp

