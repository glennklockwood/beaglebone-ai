#!/usr/bin/python3
"""
Copyright (c) 2019 Texas Instruments Incorporated - http://www.ti.com/
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Texas Instruments Incorporated nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import heapq
import logging
import subprocess

import numpy
import cv2
import flask

sys.path.append("/usr/share/ti/tidl/tidl_api")
import tidl

CAMERA_DEVICE = "/dev/video1"

# MUST make sure EVE and DSP Executor objects are not garbage collected as
# they fall out of scope - even though Python doesn't reference them, the TIDL
# runtime will transparently reference them which will trigger mystery
# segfaults if Python destroys them
EXECUTORS = {}
LABELS_CLASSES = None
EOPS = None
NUM_EOPS = None
EOP_FRAMES = None
CAMERA = None
LAST_RPT_ID = None
CONFIGURATION = None
SELECTED_ITEMS = None
APP = flask.Flask(__name__)

def filter_init():
    """Initializes the inference process
    """
    logging.info("Initializing application")

    populate_labels("/usr/share/ti/examples/tidl/classification/imagenet.txt")

    global SELECTED_ITEMS
    SELECTED_ITEMS = [
        429, # baseball
        837, # sunglasses
        504, # coffee_mug
        441, # beer_glass
        898, # water_bottle
        931, # bagel
        531, # digital_watch
        487, # cellular_telephone
        722, # ping-pong_ball
        720, # pill_bottle
    ]

    logging.info("Loading configuration")
    global CONFIGURATION
    CONFIGURATION = tidl.Configuration()
    CONFIGURATION.in_data= "/usr/share/ti/examples/tidl/test/testvecs/input/preproc_0_224x224.y"
    CONFIGURATION.network_heap_size = (20 << 20)
    CONFIGURATION.network_binary = "/usr/share/ti/examples/tidl/test/testvecs/config/tidl_models/tidl_net_imagenet_jacintonet11v2.bin"
    CONFIGURATION.param_heap_size = (3 << 20)
    CONFIGURATION.parameter_binary = "/usr/share/ti/examples/tidl/test/testvecs/config/tidl_models/tidl_param_imagenet_jacintonet11v2.bin"
    CONFIGURATION.width = 224
    CONFIGURATION.height = 224
    CONFIGURATION.channels = 3

    logging.info("Allocating I/O memory for each EOP")
    allocate_memory()

    global NUM_EOPS, EOP_FRAMES
    NUM_EOPS = len(EOPS)
    EOP_FRAMES = [None] * NUM_EOPS
    logging.info("Number of EOPs: %d", NUM_EOPS)
    logging.info("About to start ProcessFrame loop!!")

def filter_process(camera, frame_index, eop):
    """Manages work on an ExecutionObjectPipeline.

    Displays the results from a completed EOP and/or loads work on to a free
    EOP.  Notice that we load a new frame into the EOP (preprocess_image)
    before we read the output off of it (display_frame); we can do this because
    EOPs maintain separate input and output buffers.

    Args:
        camera (cv2.VideoCapture): Camera from which a frame should be read
        frame_index (int): Index of the EOP being managed
        eop (tidl.ExecutionObjectPipeline): The EOP being managed

    Returns:
        bytes or None: Part of an mjpg stream if output was read off of the EOP
    """
    do_display = False
    if eop.process_frame_wait():
        do_display = True

    # this is ProcessFrame in the C++ version
    EOP_FRAMES[frame_index] = preprocess_image(camera, eop)
    if EOP_FRAMES[frame_index] is not None:
        eop.process_frame_start_async()

    if do_display:
        return display_frame(eop, EOP_FRAMES[frame_index])

    return None

def create_execution_object_pipelines():
    """Initializes TIDL device buffers and memory.

    Identifies the TIDL devices that will be used and allocates sufficient
    resources to process video frames using all of them.  Note the C++ example
    hard-codes the number of EVEs and DSPs, but we query for all of them here.
    """
    num_eve = tidl.Executor.get_num_devices(tidl.DeviceType.EVE)
    num_dsp = tidl.Executor.get_num_devices(tidl.DeviceType.DSP)
    buffer_factor = 1

    logging.info("Running network across %d EVEs, %d DSPs", num_eve, num_dsp)

    eos = []
    eve_device_ids = set([tidl.DeviceId.ID0, tidl.DeviceId.ID1,
        tidl.DeviceId.ID2, tidl.DeviceId.ID3][0:num_eve])
    if eve_device_ids:
        EXECUTORS['eve'] = tidl.Executor(tidl.DeviceType.EVE, eve_device_ids, CONFIGURATION, 1)
        for i in range(EXECUTORS['eve'].get_num_execution_objects()):
            eos.append(EXECUTORS['eve'].at(i))

    dsp_device_ids = set([tidl.DeviceId.ID0, tidl.DeviceId.ID1,
        tidl.DeviceId.ID2, tidl.DeviceId.ID3][0:num_dsp])
    if dsp_device_ids:
        EXECUTORS['dsp'] = tidl.Executor(tidl.DeviceType.DSP, dsp_device_ids, CONFIGURATION, 1)
        for i in range(EXECUTORS['dsp'].get_num_execution_objects()):
            eos.append(EXECUTORS['dsp'].at(i))

    # initialize ExecutionObjectPipelines
    global EOPS
    EOPS = []
    for _ in range(buffer_factor):
        for i, execobj in enumerate(eos):
            EOPS.append(tidl.ExecutionObjectPipeline([execobj]))

def allocate_memory():
    """Allocates memory through the TIDL API
    """
    create_execution_object_pipelines()
    tidl.allocate_memory(EOPS)

def display_frame(eop, dst):
    """Reads and labels output from an EOP.

    Args:
        eop (tidl.ExecutionObjectPipeline): The EOP containing output
        dst (numpy.array): Original image that was loaded on to the EOP

    Returns:
        bytes or None: Part of an mjpg stream if output was read off of the EO
    """
    is_object = tf_postprocess(eop)
    if is_object is not None:
        cv2.putText(
            img=dst,
            text=LABELS_CLASSES[is_object],
            org=(15, 60),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=(0, 255, 0),
            thickness=3)

        global LAST_RPT_ID
        if LAST_RPT_ID != is_object:
            logging.info("(%d)=%s", is_object, LABELS_CLASSES[is_object])
            LAST_RPT_ID = is_object

    _, encoded_frame = cv2.imencode(".jpg", dst)
    return (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        + bytes(encoded_frame) + b'\r\n')

def tf_postprocess(eop):
    """Retrieves and decodes model output.

    Returns the label and confidence of the most-confident label identified by
    the model.  This is a vastly simplified version of the C++ example which
    uses a heap queue to sort the top values but discards all but the top.
    This also incorporates the C++ tf_expected_id function.

    Args:
        eop (tidl.ExecutionObjectPipeline): EOP from which label should be read

    Returns:
        int: Index of LABELS_CLASSES corresponding to highest-confidence
        classification
    """
    top_candidates = 3

    # values of output_array are 8 bits of confidence per label
    output_array = numpy.asarray(eop.get_output_buffer())

    # initialize priority queue - algorithically inexpensive way of figuring out
    # the top classifications
    queue = [(output_array[i], i) for i in range(top_candidates)]
    heapq.heapify(queue)

    # keep track of the largest three labels
    for i in range(top_candidates, output_array.size):
        heapq.heappushpop(queue, (output_array[i], i))

    # reverse sort top labels (largest first)
    queue.sort(key=lambda x: x[0], reverse=True)

    # iterate largest to smallest and return first match
    for _, idx in queue:
        if idx in SELECTED_ITEMS:
            return idx

    return None

def populate_labels(filename):
    """Loads ImageNet labels.

    Expects a plain-text file with one label per line and nothing else.

    Args:
        filename (str): Path to imagenet labels file
    """
    global LABELS_CLASSES
    with open(filename, "r", encoding="utf8") as ifstream:
        LABELS_CLASSES = [x.strip() for x in ifstream.readlines()]

def preprocess_image(camera, eop):
    """Reads a frame into an ExecutionObject input buffer.

    Reads a frame from the webcam, resizes it to the dimensions expected by the
    model, and rearranges its red/green/blue channel layout in memory to match
    what TIDL expects.  See the following for more information on the TIDL
    tensor format.

    https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/06_02_00_21/exports/docs/tidl_j7_01_01_00_10/ti_dl/docs/user_guide_html/md_tidl_fsg_io_tensors_format.html

    Args:
        camera (cv2.VideoCapture): Camera from which a frame should be read
        eop (tidl.ExecutionObjectPipeline): EOP to which frame will be loaded

    Returns:
        numpy.array: Unmodified frame retrieved from camera.
    """
    width = CONFIGURATION.width
    height = CONFIGURATION.height
    size = width * height

    success, frame = camera.read()
    if not success:
        return None

    # resize the frame to fit our TIDL input buffer
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # split out the channels because of the way TIDL expects images
    b_frame, g_frame, r_frame = cv2.split(resized)

    # np_arg becomes a pointer to a buffer that looks like a numpy array
    np_arg = numpy.asarray(eop.get_input_buffer())
    np_arg[0 * size:1 * size] = numpy.reshape(b_frame, size)
    np_arg[1 * size:2 * size] = numpy.reshape(g_frame, size)
    np_arg[2 * size:3 * size] = numpy.reshape(r_frame, size)

    return frame

def generate_stream(camera):
    """Generates frames to be displayed via HTTP

    Master loop that reads an image from a camera, runs it through our model,
    paints the top label on the image, and returns it as a component of an mjpg
    stream.  In the C++ version, this functionality is provided by mjpg-streamer
    itself.

    Args:
        camera (cv2.VideoCapture): Camera from which frames should be read

    Yields:
        bytes or None: Part of an mjpg stream if output was read off of the EOP
    """
    frame_index = 0
    while True:
        frame_index = (frame_index + 1) % NUM_EOPS
        eop = EOPS[frame_index]

        output_frame = filter_process(camera, frame_index, eop)
        if output_frame is not None:
            yield output_frame

@APP.route('/')
def video_feed():
    """Returns an mjpg stream of classified image frames.

    In the C++ version, this functionality is provided by mjpg-streamer itself.
    """
    return flask.Response(
        generate_stream(camera=CAMERA),
        mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    """Initializes and launches the streaming classification app.

    In the C++ version, this functionality is provided by mjpg-streamer itself.
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("Cleaning EVE and DSP heaps...")
    subprocess.call(['ti-mct-heap-check', '-c'])

    global CAMERA
    CAMERA = cv2.VideoCapture(CAMERA_DEVICE)

    filter_init()

    logging.info("http://localhost:8080/")
    try:
        APP.run(host='0.0.0.0', port=8080)
    except:
        tidl.free_memory(EOPS)
        CAMERA.release()
        raise

if __name__ == '__main__':
    main()
