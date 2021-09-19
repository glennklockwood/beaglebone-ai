# TIDL Python API Demo

This is my attempt at reimplementing the [BeagleBone AI TIDL demo][] using the
Python API for TIDL.

To get started, first download the model, parameters, and sample input by running

    ./download_datasets.sh

Then run the example:

    ./imagenet.py

This is still a work in progress.  See <https://www.glennklockwood.com/embedded/beaglebone-tidl.html> for more detailed notes.

## Core code

- `classification.py` - my implementation of the BeagleBone AI classification demo using TIDL's Python API
- `imagenet.py` - The TI example of using the Python API to do classification on
  which the BeagleBone AI demo is modeled.  Taken directly from
  <https://git.ti.com/cgit/tidl/tidl-api/tree/examples/pybind/imagenet.py>

## Helpful tools:

- `show_tidl_api.py` - Dumps the TIDL Python interfaces to stdout
- `download_datasets.sh` - Copy required example input and model+parameters into
  this directory.  They aren't stored in this git repository because they're
  big, binary, and don't change.
- `classify_url.sh` - Pass a URL to an image and this script will download and
  identify it using `imagenet.py`

[BeagleBone AI TIDL demo]: https://github.com/glennklockwood/beaglebone-ai/blob/main/classification/classification.cpp

