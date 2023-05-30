#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Expects to be invoked with argv[1] giving the path to the bitfile to load into the FPGA,
# e.g. 'test_sadc.py UnifirmTopLevel.bit'.

import errno
import re
import os.path
import optparse
import subprocess
import sys
import time
import numpy as np
import random as rdm

import samna
from samna.dynapse2 import *



def readable(file):
    try:
        f = open(file, mode='rb', buffering=0)
    except OSError:
        return False
    f.close()
    return True


def bus_and_dev_from_vid_pid_string(vid_pid_string):
    print(vid_pid_string)
    cp = subprocess.run(['lsusb', '-d', vid_pid_string], stdout=subprocess.PIPE,
                        universal_newlines=True)
    if cp.returncode != 0:
        print('Dynap-se2 not found')
        exit(errno.ENODEV)

    regexp = re.compile('Bus (\d{3}) Device (\d{3}): ID [0-9a-f]{4}:[0-9a-f]{4} *.')
    m = regexp.match(cp.stdout)
    if m is None:
        print('Unexpected output from lsusb: "' + cp.stdout + '"')
        exit(errno.EPROTO)

    return (int(m.group(1)), int(m.group(2)))


def connect(device, n_chips, samna_node, sender_endpoint, receiver_endpoint, node_id, interpreter_id):
    assert(node_id != interpreter_id)

    if device == 'devboard':
        vid_pid_string = '%04x:%04x' % samna_node.get_dynapse2_dev_board_vid_and_pid()
    if device == 'stack':
        vid_pid_string = '%04x:%04x' % samna_node.get_dynapse2_stack_vid_and_pid()
    bus, dev = bus_and_dev_from_vid_pid_string(vid_pid_string)
    print('Bus %03d Device %03d: ID %s' % (bus, dev, vid_pid_string))
    if device == 'devboard':
        samna_node.open_dynapse2_dev_board(bus, dev)
    if device == 'stack':
        samna_node.open_dynapse2_stack(bus, dev, n_chips)

    samna.setup_local_node(receiver_endpoint, sender_endpoint, interpreter_id)
    samna.open_remote_node(node_id, "device_node")

    return samna.device_node


def set_parameter(parameters, name, coarse, fine):
    parameter = parameters[name]
    parameter.coarse_value = coarse
    parameter.fine_value = fine
    # print(name, parameter)
    return


def main():
    parser = optparse.OptionParser()
    parser.set_usage("Usage: test_sadc.py [options] bitfile [number_of_chips]")
    parser.add_option("-d", "--devboard", action="store_const", const="devboard", dest="device",
        help="use first XEM7360 found together with DYNAP-SE2 DevBoard")
    parser.add_option("-s", "--stack", action="store_const", const="stack", dest="device",
        help="use first XEM7310 found together with DYNAP-SE2 Stack board(s)")
    parser.set_defaults(device="stack")
    opts, args = parser.parse_args()

    if len(args) == 0:
        print('No bitfile specified')
        exit(errno.EINVAL)

    if len(args) > 2:
        print('Too many arguments')
        exit(errno.E2BIG)

    bitfile = args[0]
    if len(args) == 2:
        number_of_chips = int(args[1])
    else:
        number_of_chips = 1

    if not os.path.isfile(bitfile):
        print('Bitfile %s not found' % bitfile)
        exit(errno.ENOENT)

    if not readable(bitfile):
        print('Cannot read %s' % bitfile)
        exit(errno.EACCES)

    receiver_endpoint = "tcp://0.0.0.0:33335"
    sender_endpoint = "tcp://0.0.0.0:33336"
    node_id = 1
    interpreter_id = 2
    samna_node = samna.SamnaNode(sender_endpoint, receiver_endpoint, node_id)
    remote = connect(opts.device, number_of_chips, samna_node, sender_endpoint, receiver_endpoint, node_id, interpreter_id)

    if opts.device == 'devboard':
        board = remote.Dynapse2DevBoard
    if opts.device == 'stack':
        board = remote.Dynapse2Stack

    if not board.configure_opal_kelly(bitfile):
        print('Failed to configure Opal Kelly')
        exit(errno.EIO)

    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)

    # dump initial configurations
    config = model.get_configuration()
    model.apply_configuration(config)
    time.sleep(1)

    # select core in {0, 1, 2, 3}
    i_core = 0

    ## config neural core

    # set neuron R1LUT for monitoring
    neuron_monitor_dest = Dynapse2Destination()
    neuron_monitor_dest.core = [False, False, False, True]
    neuron_monitor_dest.x_hop = -1
    neuron_monitor_dest.y_hop = 0
    neuron_non_dest =  Dynapse2Destination()
    neuron_non_dest.core = [False, False, False, False]
    neuron_non_dest.x_hop = -1
    neuron_non_dest.y_hop = 0
    for i in range(256):
        neuron_monitor_dest.tag = i
        config.chips[0].cores[i_core].neurons[i].destinations = [neuron_monitor_dest] + [neuron_non_dest] * 3
    model.apply_configuration(config) #current consumption lowers here
    time.sleep(0.1)

    # Set neuron biases
    config.chips[0].cores[i_core].monitored_neuron = 0x00
    set_parameter(config.chips[0].global_parameters, 'R2R_BUFFER_AMPB', 5, 255)
    set_parameter(config.chips[0].global_parameters, 'R2R_BUFFER_CCB', 5, 255)
    set_parameter(config.chips[0].shared_parameters01, 'LBWR_VB_P', 5, 255)
    set_parameter(config.chips[0].shared_parameters01, 'PG_BUF_N', 5, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SOIF_DC_P', 0, 0)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DENM_NMREV_N', 1, 126)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SOIF_REFR_N', 4, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SOIF_LEAK_N', 4, 50)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SOIF_GAIN_N', 3, 150)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SOIF_SPKTHR_P', 3, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DEAM_ETAU_P', 4, 30)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DEAM_EGAIN_P', 4, 120)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DEAM_ITAU_P', 3, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DEAM_IGAIN_P', 0, 0)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DENM_ETAU_P', 0, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DENM_EGAIN_P', 0, 0)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DENM_ITAU_P', 3, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DENM_IGAIN_P', 0, 0)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DEGA_ITAU_P', 2, 115)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DEGA_IGAIN_P', 0, 0)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DESC_ITAU_P', 4, 30)
    set_parameter(config.chips[0].cores[i_core].parameters, 'DESC_IGAIN_P', 4, 90)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYPD_DLY0_P', 3, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYPD_DLY1_P', 3, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYPD_DLY2_P', 3, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYSA_VRES_N', 5, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYSA_VB_P', 5, 255)
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYAM_W0_P', 4, 80) # neuron group 1 excitatory weight
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYAM_W1_P', 4, 80) # neuron group 1 inhibitory weight
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYAM_W2_P', 4, 80) # neuron group 2 excitatory weight
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYAM_W3_P', 4, 80) # neuron group 2 inhibitory weight
    set_parameter(config.chips[0].cores[i_core].parameters, 'SYPD_EXT_N', 3, 100)
    config.chips[0].cores[i_core].neuron_monitoring_on = True
    model.apply_configuration(config) #current consumption lowers here
    time.sleep(0.1)

    ## Configure the BioAmp Config and BioAmp Routing registers
    bioamps_features = config.chips[0].bioamps

    # Config BAR register: set the route using Dynapse2Destination class
    adm_monitor_dest = Dynapse2Destination()
    adm_monitor_dest.core = [True, False, False, False]
    adm_monitor_dest.tag = 5
    adm_monitor_dest.x_hop = -1
    adm_monitor_dest.y_hop = 0
    adm_connect_dest =  Dynapse2Destination()
    adm_connect_dest.core = [True, False, False, False]
    adm_connect_dest.tag = 5
    adm_connect_dest.x_hop = 0
    adm_connect_dest.y_hop = 0
    config.chips[0].bioamps.route = [adm_monitor_dest, adm_connect_dest]
    model.apply_configuration(config) #current consumption lowers here
    time.sleep(0.1)

    # Configure BAC register
    config.chips[0].bioamps.monitor_channel_oauc = True
    config.chips[0].bioamps.monitor_channel_oruc = True
    config.chips[0].bioamps.monitor_channel_osuc = True
    config.chips[0].bioamps.monitor_channel_qfruc = True
    config.chips[0].bioamps.monitor_channel_thdc = True
    config.chips[0].bioamps.monitor_channel_thuc = True
    config.chips[0].bioamps.param_gen2_powerdown = False
    config.chips[0].bioamps.gain = 3
    model.apply_configuration(config)
    time.sleep(0.1)

    ## Set channels common parameters
    set_parameter(bioamps_features.common_parameters, "PG_BUF_N", 5, 255)
    set_parameter(bioamps_features.common_parameters, "BAVBSS", 5, 46)
    set_parameter(bioamps_features.common_parameters, "BAVBIAS", 5, 36)
    set_parameter(bioamps_features.common_parameters, "BAVCAS", 5, 116)
    set_parameter(bioamps_features.common_parameters, "BAVCASC", 5, 116)
    set_parameter(bioamps_features.common_parameters, "BCCB", 5, 106)
    set_parameter(bioamps_features.common_parameters, "BAMPB", 4,  45)
    set_parameter(bioamps_features.common_parameters, "BFVBIAS", 1, 34)
    set_parameter(bioamps_features.common_parameters, "BF1VG", 0, 198)
    set_parameter(bioamps_features.common_parameters, "BF1VF", 0, 198)
    set_parameter(bioamps_features.common_parameters, "BF1VQ", 0, 116)
    set_parameter(bioamps_features.common_parameters, "BADCVBIAS", 5, 30)
    set_parameter(bioamps_features.common_parameters, "BBVBIAS", 3, 34)
    set_parameter(bioamps_features.common_parameters, "BADCREFRACTORY",5,255)
    model.apply_configuration(config) #current consumption raises here
    time.sleep(0.1)

    # Set the individual channel biases for the ADM
    for channel in range(8):
        set_parameter(bioamps_features.channel_parameters[channel], "BADCTHU", 2, 80)
        set_parameter(bioamps_features.channel_parameters[channel], "BADCTHUC", 2, 80)
        set_parameter(bioamps_features.channel_parameters[channel], "BADCTHD", 2, 40)
        set_parameter(bioamps_features.channel_parameters[channel], "BADCTHDC", 2, 40)
        set_parameter(bioamps_features.channel_parameters[channel], "BF2VG", 0, 40)
        set_parameter(bioamps_features.channel_parameters[channel], "BF2VF", 0, 40)
        set_parameter(bioamps_features.channel_parameters[channel], "BF2VQ", 0, 40)
        set_parameter(bioamps_features.channel_parameters[channel], "BF3VG", 0, 40)
        set_parameter(bioamps_features.channel_parameters[channel], "BF3VF", 0, 40)
        set_parameter(bioamps_features.channel_parameters[channel], "BF3VQ", 0, 40)
    model.apply_configuration(config)
    time.sleep(0.1)

    ## set connections
    input_adms = 4
    adm_id_offset = 4
    input_virtual_neuron_id = np.arange(input_adms) + adm_id_offset + (adm_connect_dest.tag << 6)
    input_channels = 8

    cam_non = Dynapse2Synapse()
    cam_non.weight = [False, False, False, False]
    cam_non.dendrite = Dendrite.none
    cam_non.tag = 0

    # Set post-neurons with up excitatory and down inhibitory connections
    post_neurons_to_stimulate = np.arange(0, 128)
    # For each post-synaptic neuron
    for i, i_neu in enumerate(post_neurons_to_stimulate):
        # For each virtual neiron
        cams = []
        num_cams = 0
        for j, j_adm  in enumerate(input_virtual_neuron_id):
            # If input neuron is 1 or 3 (UP channels) connection is excitatory
            cam_event = [Dynapse2Synapse()] * input_channels
            if j % 2 == 0:
                synapse_type = Dendrite.ampa
                num_cams = rdm.choice([1,2])
                weight = [True, False, False, False]
            else:
                synapse_type = Dendrite.shunt
                weight = [False, True, False, False]
            # For each cam that you want to use for the specific input
            for k in range(input_channels):
                cam_event[k].dendrite = synapse_type
                cam_event[k].weight = weight
                cam_event[k].tag = k * 8 + j_adm
            cams += cam_event * num_cams
        config.chips[0].cores[i_core].neurons[i_neu].synapses = cams + [cam_non] * (64 - len(cams))
    model.apply_configuration(config) #current consumption lowers here
    time.sleep(1)

    # Set post-neurons with down excitatory and up inhibitory connections
    post_neurons_to_stimulate = np.arange(128, 256)

    # For each post-synaptic neuron
    for i, i_neu in enumerate(post_neurons_to_stimulate):
        # For each virtual neiron
        cams = []
        num_cams = 0
        for j, j_adm  in enumerate(input_virtual_neuron_id):
            # If input neuron is 1 or 3 (UP channels) connection is excitatory
            cam_event = [Dynapse2Synapse()] * input_channels
            if j % 2 == 0:
                synapse_type = Dendrite.shunt
                num_cams = rdm.choice([1,2])
                weight = [False, False, False, True]
            else:
                synapse_type = Dendrite.ampa
                weight = [False, False, True, False]
            # For each cam that you want to use for the specific input
            for k in range(input_channels):
                cam_event[k].dendrite = synapse_type
                cam_event[k].weight = weight
                cam_event[k].tag = k * 8 + j_adm
            cams += cam_event * num_cams
        config.chips[0].cores[i_core].neurons[i_neu].synapses = cams + [cam_non] * (64 - len(cams))
    model.apply_configuration(config) #current consumption lowers here
    time.sleep(1)

    # command line raster plot, make terminal at least 266 characters wide
    adm_chars = ['L', 'l', 'A', 'a', 'R', 'r', 'F', 'f']
    flag = True
    while True:
        counter = 0
        while counter < 100000:
            for ev in board.output_read():
                if flag:
                    addr = (ev >> 12) % 256
                    if (ev >> 20) % 16 == adm_monitor_dest.tag:
                        # display ADM events, UPPER CASE = Up, lower case = down
                        # L/l = low-pass filtered
                        # A/a = amplifier
                        # R/r = ripple band
                        # F/f = fast ripple band
                        print(" "*((addr >> 3) * 2 + (addr % 2)) + adm_chars[addr % 8])
                    else:
                        # display neural activity
                        print(" "*(addr + 10) + "*")
                    # print('%08x' % ev)
                flag = not flag
                counter = 0
            time.sleep(0.00001)
            counter += 1
        model.reset(ResetType.LogicReset, 0b1)


if __name__ == '__main__':
    main()
