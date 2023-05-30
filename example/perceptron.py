import time
import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import set_parameter, clear_srams
from lib.dynapse2_network import Network
from lib.dynapse2_spikegen import get_fpga_time, send_virtual_events, poisson_gen
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *

import random

from samna.dynapse2 import *


def perceptron_xor(board, profile_path, number_of_chips):
    # your code starts here
    model = board.get_model()
    model.reset(ResetType.PowerCycle, (1 << number_of_chips) - 1)
    time.sleep(1)

    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    neuron = 3
    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 0, 9)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 1, 255)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 255)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set synapse parameters  -- enabled AM and SC
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 2, 160)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_ITAU_P', 1, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', 4, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 0, 0)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 3, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', 4, 160)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', 4, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 4, 80)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips)
    input1 = network.add_virtual_group(size=4)
    input2 = network.add_virtual_group(size=4)
    hidden1 = network.add_group(chip=0, core=0, size=32)
    hidden2 = network.add_group(chip=0, core=0, size=32)
    output1 = network.add_group(chip=0, core=0, size=32)
    for input_group in [input1, input2]:
        network.add_connection(source=input_group, target=hidden1, probability=0.5,
                               dendrite=Dendrite.ampa, weight=[False, True, False, False], repeat=8)
        network.add_connection(source=input_group, target=hidden2, probability=0.5,
                               dendrite=Dendrite.nmda, weight=[False, True, False, False], repeat=8)
    network.add_connection(source=hidden1, target=output1, probability=0.5,
                           dendrite=Dendrite.ampa, weight=[False, False, True, False])
    network.add_connection(source=hidden2, target=output1, probability=0.5,
                           dendrite=Dendrite.shunt, weight=[False, False, False, True])
    network.connect()
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print(hidden1.neurons)

    print("\nAll configurations done!\n")

    input_events = isi_gen(virtual_group, neurons, timestamps)
    input_events = poisson_gen(start=0, duration=1000000, virtual_groups=[input1, input2], rates=[200] * 2) + \
                   poisson_gen(start=1000000, duration=1000000, virtual_groups=[input1, input2], rates=[100] * 2)

    ts = get_fpga_time(board=board) + 100000
    send_virtual_events(board=board, virtual_events=input_events, offset=ts, min_delay=100000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)
