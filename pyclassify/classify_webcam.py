#!/usr/bin/python3

# Copyright (c) 2019 Texas Instruments Incorporated - http://www.ti.com/
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# * Neither the name of Texas Instruments Incorporated nor the
# names of its contributors may be used to endorse or promote products
# derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

""" Process each frame using a single ExecutionObject.
    Increase throughput by using multiple ExecutionObjects.
"""

import os
import sys
import argparse
import json
import heapq
import logging
import numpy as np
import cv2
import flask

sys.path.append("/usr/share/ti/tidl/tidl_api")
import tidl

# MUST make sure EVE and DSP Executor objects are not garbage collected as
# they fall out of scope - even though Python doesn't reference them, the TIDL
# runtime will transparently reference them which will trigger mystery
# segfaults if Python destroys them
EXECUTORS = {}
EOPS = None
CAMERA = None
CONFIGURATION = None
APP = flask.Flask(__name__)

@APP.route('/')
def video_feed():
    global CAMERA, CONFIGURATION
    return flask.Response(
        stream_camera(
            camera=CAMERA,
            num_eve=4,
            num_dsp=2,
            configuration=CONFIGURATION),
        mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    """Read the configuration and run the network"""
    global CAMERA, CONFIGURATION

    logging.basicConfig(level=logging.INFO)


    CONFIGURATION = tidl.Configuration()
    CONFIGURATION.read_from_file("imagenet.conf")

    CAMERA = cv2.VideoCapture("/dev/video1")

    APP.run(host='0.0.0.0')

def init_tidl(configuration, num_eve, num_dsp, pipeline_depth):
    global EXECUTORS
    tot_eve = tidl.Executor.get_num_devices(tidl.DeviceType.EVE)
    tot_dsp = tidl.Executor.get_num_devices(tidl.DeviceType.DSP)
    if num_eve < 0 or num_eve > tot_eve:
        raise ValueError(f"Must use between 0 and {tot_eve:d} EVEs")
    if num_dsp < 0 or num_dsp > tot_dsp:
        raise ValueError(f"Must use between 0 and {tot_dsp:d} DSPs")
    if tot_eve == 0 and tot_dsp == 0:
        raise RuntimeError('No TIDL API capable devices available')

    logging.info(
        'Running network across {} EVEs, {} DSPs'.format(num_eve, num_dsp))

    dsp_device_ids = set([tidl.DeviceId.ID0, tidl.DeviceId.ID1,
                          tidl.DeviceId.ID2, tidl.DeviceId.ID3][0:num_dsp])
    eve_device_ids = set([tidl.DeviceId.ID0, tidl.DeviceId.ID1,
                          tidl.DeviceId.ID2, tidl.DeviceId.ID3][0:num_eve])

    # Heap sizes for this network determined using Configuration.showHeapStats
    configuration.param_heap_size = (3 << 20)
    configuration.network_heap_size = (20 << 20)

    logging.info('TIDL API: performing one time initialization ...')

    # Collect all EOs from EVE and DSP executors
    eos = []
    if eve_device_ids:
        EXECUTORS['eve'] = tidl.Executor(tidl.DeviceType.EVE, eve_device_ids, configuration, 1)
        for i in range(EXECUTORS['eve'].get_num_execution_objects()):
            eos.append(EXECUTORS['eve'].at(i))

    if dsp_device_ids:
        EXECUTORS['dsp'] = tidl.Executor(tidl.DeviceType.DSP, dsp_device_ids, configuration, 1)
        for i in range(EXECUTORS['dsp'].get_num_execution_objects()):
            eos.append(EXECUTORS['dsp'].at(i))

    # initialize ExecutionObjectPipelines
    eops = []
    for j in range(pipeline_depth):
        for i, eo in enumerate(eos):
            eops.append(tidl.ExecutionObjectPipeline([eo]))

    tidl.allocate_memory(eops)

    return eops

def device_name_to_eo(device_name):
    """Translates device name to an ExecutionObject.
    
    Receives the string output from ExecutionObjectPipeline.get_device_name(),
    finds the ExecutionObject with the same name, and returns it.  Expects only
    device_names of the form EVE# or DSP#.

    Args:
        device_name (str): EVE0, EVE1, EVE2, EVE3, DSP0, or DSP1.

    Returns:
        tidl.ExecutionObject or None.
    """
    device_type = device_name[0:3].lower()
    executor = EXECUTORS.get(device_type)
    if not executor:
        raise RuntimeError("Executors not yet initialized")

    for i in range(executor.get_num_execution_objects()):
        this_eo = executor.at(i)
        this_device = this_eo.get_device_name()
        if this_device.lower() == device_name.lower():
            return this_eo 

    return None


def stream_camera(camera, num_eve, num_dsp, configuration, labels_file='imagenet_objects.json' , pipeline_depth=2):
    """ Run the network on the specified device type and number of devices"""
    global EOPS

    if not EOPS:
        EOPS = init_tidl(
            configuration=configuration,
            num_eve=num_eve,
            num_dsp=num_dsp,
            pipeline_depth=pipeline_depth)
    eops = EOPS

    # initialize TIDL, devices, etc
    num_eops = len(eops)
    frames = [None] * num_eops

    # open labels file
    with open(labels_file) as json_file:
        labels_data = json.load(json_file)

    # initialize camera

    frame_index = 0
    while True:
        frame_index = frame_index % num_eops
        eop = eops[frame_index]

        # once an EOP becomes available, process its output
        if eop.process_frame_wait():
            labels = process_output(eop, labels_data)
            if len(labels) > 0:
                cv2.putText(
                    img=frames[frame_index],
                    text="{} {}".format(*labels[0]),
                    org=(15, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=3)
            _, encoded_frame = cv2.imencode(".jpg", frames[frame_index])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                + bytes(encoded_frame) + b'\r\n')

        # load and process next frame
        frames[frame_index] = read_frame(camera, eop, configuration)
        if frames[frame_index] is not None:
            eop.process_frame_start_async()

    tidl.free_memory(eops)

def read_frame(camera, eop, configuration):
    """Read a frame into the ExecutionObject input buffer"""

    # Read into the EO's input buffer
    arg_info = eop.get_input_buffer()
    np_arg = np.asarray(arg_info)

    success, frame = camera.read()
    if not success:
        return None

    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    b_frame, g_frame, r_frame = cv2.split(resized)
    np_arg[0*224*224:1*224*224] = np.reshape(b_frame, 224*224)
    np_arg[1*224*224:2*224*224] = np.reshape(g_frame, 224*224)
    np_arg[2*224*224:3*224*224] = np.reshape(r_frame, 224*224)

    return frame

def process_output(eop, labels_data):
    """Display the inference result using labels."""

    # keep top k predictions in heap
    k = 5

    # output predictions with probability of 10/255 or higher
    threshold = 10

    out_buffer = eop.get_output_buffer()
    output_array = np.asarray(out_buffer)

    k_heap = []
    for i in range(k):
        heapq.heappush(k_heap, (output_array[i], i))

    for i in range(k, out_buffer.size()):
        if output_array[i] > k_heap[0][0]:
            heapq.heappushpop(k_heap, (output_array[i], i))

    k_sorted = []
    for i in range(k):
        k_sorted.insert(0, heapq.heappop(k_heap))

    results = []
    for i in range(k):
        if k_sorted[i][0] > threshold:
            results.append((
                labels_data['objects'][k_sorted[i][1]]['label'],
                k_sorted[i][0]/255.0*100))
    return results

if __name__ == '__main__':
    main()
