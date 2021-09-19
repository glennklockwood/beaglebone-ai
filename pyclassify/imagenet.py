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
import time
import argparse
import json
import heapq
import logging
import numpy as np
import cv2

sys.path.append("/usr/share/ti/tidl/tidl_api")
import tidl

def main():
    """Read the configuration and run the network"""

    parser = argparse.ArgumentParser(description='Run the imagenet network on input image.')
    parser.add_argument('input_file', nargs="+", type=str, help='input image file (that OpenCV can read)')
    parser.add_argument("-p", "--pipeline-depth", type=int, default=2, help="pipeline depth (default: 2)")
    parser.add_argument("-e", "--num-eve", type=int, default=1, help="number of EVEs to use")
    parser.add_argument("-d", "--num-dsp", type=int, default=0, help="number of DSPs to use")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    config_file = "imagenet.conf"
    labels_file = 'imagenet_objects.json'

    configuration = tidl.Configuration()
    configuration.read_from_file(config_file)

    num_eve = tidl.Executor.get_num_devices(tidl.DeviceType.EVE)
    num_dsp = tidl.Executor.get_num_devices(tidl.DeviceType.DSP)
    if args.num_eve < 0 or args.num_eve > num_eve:
        raise ValueError(f"Must use between 0 and {num_eve:d} EVEs")
    if args.num_dsp < 0 or args.num_dsp > num_dsp:
        raise ValueError(f"Must use between 0 and {num_dsp:d} DSPs")
    if num_eve == 0 and num_dsp == 0:
        raise RuntimeError('No TIDL API capable devices available')

    # use 1 EVE or DSP since input is a single image
    # If input is a stream of images, feel free to use all EVEs and/or DSPs
#    if num_eve > 0:
#        num_eve = 1
#        num_dsp = 0
#    else:
#        num_dsp = 1

    run(args.input_file, args.num_eve, args.num_dsp, configuration, labels_file, args.pipeline_depth)

    return




def run(input_files, num_eve, num_dsp, configuration, labels_file, pipeline_depth=2):
    """ Run the network on the specified device type and number of devices"""

    logging.info('Running network across {} EVEs, {} DSPs'.format(num_eve,
                                                                  num_dsp))

    dsp_device_ids = set([tidl.DeviceId.ID0, tidl.DeviceId.ID1,
                          tidl.DeviceId.ID2, tidl.DeviceId.ID3][0:num_dsp])
    eve_device_ids = set([tidl.DeviceId.ID0, tidl.DeviceId.ID1,
                          tidl.DeviceId.ID2, tidl.DeviceId.ID3][0:num_eve])

    # Heap sizes for this network determined using Configuration.showHeapStats
    configuration.param_heap_size = (3 << 20)
    configuration.network_heap_size = (20 << 20)

    logging.info('TIDL API: performing one time initialization ...')
    time0 = time.time()

    # Collect all EOs from EVE and DSP executors
    eos = []

    if eve_device_ids:
        eve = tidl.Executor(tidl.DeviceType.EVE, eve_device_ids, configuration, 1)
        for i in range(eve.get_num_execution_objects()):
            eos.append(eve.at(i))

    if dsp_device_ids:
        dsp = tidl.Executor(tidl.DeviceType.DSP, dsp_device_ids, configuration, 1)
        for i in range(dsp.get_num_execution_objects()):
            eos.append(dsp.at(i))
    num_eos = len(eos)

    eops = []
    eop_map = []
    for j in range(pipeline_depth):
        for i in range(num_eos):
            eops.append(tidl.ExecutionObjectPipeline([eos[i]]))
            eop_map.append(eos[i])

    tidl.allocate_memory(eops)
    num_eops = len(eops)
    logging.info("One-time initialization took {:.4f} seconds".format(time.time() - time0))

    # open labels file
    with open(labels_file) as json_file:
        labels_data = json.load(json_file)

    configuration.num_frames = len(input_files)
    logging.info(f'TIDL API: processing {configuration.num_frames} input frames ...')

    time0 = time.time()
    for frame_index in range(configuration.num_frames + num_eops):
        # assign each frame to an EOP
        eop = eops[frame_index % num_eops]

        # once an EOP becomes available, process its output
        if eop.process_frame_wait():
            print("Input: {}".format(input_files[eop.get_frame_index()]))
            for oidx, output in enumerate(process_output(eop, labels_data)):
                print('{}: {},   prob = {:5.2f}%'.format(
                    oidx+1, output[0], output[1]))
            logging.info("Took {:.1f} ms on {}".format(
                eop_map[frame_index % num_eops].get_process_time_in_ms(),
                eop.get_device_name()))

        # read the next frame and enqueue its processing
        if frame_index < configuration.num_frames:
            configuration.in_data = input_files[frame_index]
        else:
            continue

        if read_frame(eop, frame_index, configuration):
            print("Launching {} on {}".format(configuration.in_data, eop.get_device_name()))
            eop.process_frame_start_async()

    timef = time.time() - time0
    print("Processed {:d} frames in {:.4f} seconds ({:.1f} fps)".format(
        configuration.num_frames,
        timef,
        configuration.num_frames / timef))
    tidl.free_memory(eops)

def read_frame(eo, frame_index, configuration):
    """Read a frame into the ExecutionObject input buffer"""

    # this seems sloppy
    if frame_index >= configuration.num_frames:
        return False

    # Read into the EO's input buffer
    arg_info = eo.get_input_buffer()
    np_arg = np.asarray(arg_info)

    img = cv2.imread(configuration.in_data)
    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    b_frame, g_frame, r_frame = cv2.split(resized)
    np_arg[0*224*224:1*224*224] = np.reshape(b_frame, 224*224)
    np_arg[1*224*224:2*224*224] = np.reshape(g_frame, 224*224)
    np_arg[2*224*224:3*224*224] = np.reshape(r_frame, 224*224)

    eo.set_frame_index(frame_index)

    return True

def process_output(eo, labels_data):
    """Display the inference result using labels."""

    # keep top k predictions in heap
    k = 5

    # output predictions with probability of 10/255 or higher
    threshold = 10

    out_buffer = eo.get_output_buffer()
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
