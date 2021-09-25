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
import time
import json
import logging
import subprocess

import numpy
import cv2
import flask

sys.path.append("/usr/share/ti/tidl/tidl_api")
import tidl

# MUST make sure EVE and DSP Executor objects are not garbage collected as
# they fall out of scope - even though Python doesn't reference them, the TIDL
# runtime will transparently reference them which will trigger mystery
# segfaults if Python destroys them
EXECUTORS = {}
LABELS = {}
EOPS = None
CAMERA = None
CONFIGURATION = None
APP = flask.Flask(__name__)

FPS_QUEUE_LEN = 10 # calculate fps over this many seconds
MIN_CONFIDENCE = 25 # don't display labels if confidence is lower than this (%)

@APP.route('/')
def video_feed():
    """Returns an mjpg stream of classified image frames.
    """
    return flask.Response(
        stream_camera(camera=CAMERA),
        mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    """Initializes and launches the streaming classification app.
    """
    global CAMERA, CONFIGURATION, LABELS, EOPS

    logging.basicConfig(level=logging.INFO)

    logging.info("Cleaning EVE and DSP heaps...")
    subprocess.call(['ti-mct-heap-check', '-c'])

    CONFIGURATION = tidl.Configuration()
    CONFIGURATION.read_from_file("imagenet.conf")

    CAMERA = cv2.VideoCapture("/dev/video1")

    EOPS = init_tidl(configuration=CONFIGURATION, pipeline_depth=2)

    with open("imagenet_objects.json", "r", encoding="utf8") as json_file:
        LABELS = json.load(json_file)

    try:
        APP.run(host='0.0.0.0')
    except:
        tidl.free_memory(EOPS)
        CAMERA.release()
        raise

def init_tidl(configuration, pipeline_depth):
    """Initializes TIDL devices, memory, and state.

    Identifies the TIDL devices that will be used and allocates sufficient
    resources to process video frames using all of them.
    """
    global EOPS

    num_eve = tidl.Executor.get_num_devices(tidl.DeviceType.EVE)
    num_dsp = tidl.Executor.get_num_devices(tidl.DeviceType.DSP)

    logging.info("Running network across %d EVEs, %d DSPs", num_eve, num_dsp)

    configuration.param_heap_size = (3 << 20)
    configuration.network_heap_size = (20 << 20)

    # Collect all EOs from EVE and DSP executors
    eos = []
    eve_device_ids = set([tidl.DeviceId.ID0, tidl.DeviceId.ID1,
        tidl.DeviceId.ID2, tidl.DeviceId.ID3][0:num_eve])
    if eve_device_ids:
        EXECUTORS['eve'] = tidl.Executor(tidl.DeviceType.EVE, eve_device_ids, configuration, 1)
        for i in range(EXECUTORS['eve'].get_num_execution_objects()):
            eos.append(EXECUTORS['eve'].at(i))

    dsp_device_ids = set([tidl.DeviceId.ID0, tidl.DeviceId.ID1,
        tidl.DeviceId.ID2, tidl.DeviceId.ID3][0:num_dsp])
    if dsp_device_ids:
        EXECUTORS['dsp'] = tidl.Executor(tidl.DeviceType.DSP, dsp_device_ids, configuration, 1)
        for i in range(EXECUTORS['dsp'].get_num_execution_objects()):
            eos.append(EXECUTORS['dsp'].at(i))

    # initialize ExecutionObjectPipelines
    EOPS = []
    for _ in range(pipeline_depth):
        for i, execobj in enumerate(eos):
            EOPS.append(tidl.ExecutionObjectPipeline([execobj]))

    tidl.allocate_memory(EOPS)

    return EOPS

def stream_camera(camera):
    """Classifies image from a webcam, labels them, and yields them.

    Master loop that reads an image from a camera, runs it through our model,
    paints the top label on the image, and returns it as a component of an mjpg
    stream.
    """
    num_eops = len(EOPS)
    logging.info('Processing using %d ExecutionObjects', num_eops)
    frames = [None] * num_eops
    timing_start = time.time()
    tot_frames = 0

    frame_index = 0
    while True:
        frame_index = (frame_index + 1) % num_eops
        eop = EOPS[frame_index]

        # once an EOP becomes available, process its output
        if eop.process_frame_wait():
            # calculate framerate
            now = time.time()
            elapsed = now - timing_start
            if elapsed > FPS_QUEUE_LEN:
                fps = tot_frames / elapsed
                logging.info("Inference rate: %.0f fps", fps)
                tot_frames = 0
                timing_start = now

            # get the most confidence classification
            label, confidence = get_classification(eop, LABELS)
            tot_frames += 1
            if confidence >= MIN_CONFIDENCE:
                # overlay the classification and confidence on the image
                text = label.replace("_", " ") + f" {confidence:.0f}%"
                cv2.putText(
                    img=frames[frame_index],
                    text=text,
                    org=(15, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=3)

            # send the next frame to the mjpg stream
            _, encoded_frame = cv2.imencode(".jpg", frames[frame_index])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                + bytes(encoded_frame) + b'\r\n')

        # load and start processing the next frame
        frames[frame_index] = read_frame(camera, eop)
        if frames[frame_index] is not None:
            eop.process_frame_start_async()

def read_frame(camera, eop):
    """Reads a frame into an ExecutionObject input buffer.

    Reads a frame from the webcam, resizes it to the dimensions expected by the
    model, and rearranges its red/green/blue channel layout in memory to match
    what TIDL expects.  See the following for more information on the TIDL
    tensor format.

    https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/06_02_00_21/exports/docs/tidl_j7_01_01_00_10/ti_dl/docs/user_guide_html/md_tidl_fsg_io_tensors_format.html

    Returns:
        numpy.array: Unmodified frame retrieved from webcam.
    """

    success, frame = camera.read()
    if not success:
        return None

    # resize the frame to fit our TIDL input buffer
    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # split out the channels because of the way TIDL expects images
    b_frame, g_frame, r_frame = cv2.split(resized)

    # np_arg becomes a pointer to a buffer that looks like a numpy array
    np_arg = numpy.asarray(eop.get_input_buffer())
    np_arg[0*224*224:1*224*224] = numpy.reshape(b_frame, 224*224)
    np_arg[1*224*224:2*224*224] = numpy.reshape(g_frame, 224*224)
    np_arg[2*224*224:3*224*224] = numpy.reshape(r_frame, 224*224)

    return frame

def get_classification(eop, labels_data):
    """Retrieves and decodes model output.

    Returns the label and confidence of the most-confident label identified by
    the model.

    Returns:
        tuple of str, float: label and confidence expressed between 0 and 100.
    """
    # values of output_array are 8 bits of confidence per label
    output_array = numpy.asarray(eop.get_output_buffer())
    best_label = numpy.argmax(output_array)
    return (labels_data['objects'][best_label]['label'], 100 * output_array[best_label] / 255)

if __name__ == '__main__':
    main()
