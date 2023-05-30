import itertools
import time
import sys
import os

sys.path.append(os.getcwd() + '/..')

import random

from lib.dynapse2_util import set_parameter, clear_srams
from lib.dynapse2_obj import *
from lib.dynapse2_spikegen import *
from lib.dynapse2_raster import *
from lib.dynapse2_network import Network

from samna.dynapse2 import *


def config_parameters(myConfig):
    # set neuron parameters
    set_parameter(myConfig.chips[0].shared_parameters01, "PG_BUF_N", 0, 100)
    set_parameter(myConfig.chips[0].shared_parameters23, "PG_BUF_N", 0, 100)
    set_parameter(myConfig.chips[0].cores[0].parameters, "SOIF_GAIN_N", 3, 80)
    set_parameter(myConfig.chips[0].cores[1].parameters, "SOIF_GAIN_N", 3, 80)
    set_parameter(myConfig.chips[0].cores[0].parameters, "SOIF_LEAK_N", 2, 30)
    set_parameter(myConfig.chips[0].cores[1].parameters, "SOIF_LEAK_N", 2, 30)
    set_parameter(myConfig.chips[0].cores[0].parameters, "SOIF_REFR_N", 2, 60)
    set_parameter(myConfig.chips[0].cores[1].parameters, "SOIF_REFR_N", 3, 60)
    set_parameter(myConfig.chips[0].cores[1].parameters, "SOIF_DC_P", 4, 80)
    set_parameter(myConfig.chips[0].cores[0].parameters, "SOIF_SPKTHR_P", 3, 254)
    set_parameter(myConfig.chips[0].cores[1].parameters, "SOIF_SPKTHR_P", 3, 254)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'DEAM_ETAU_P', 4, 40)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'DEAM_EGAIN_P', 5, 40)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'DENM_ETAU_P', 2, 20)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'DENM_EGAIN_P', 2, 160)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'DEGA_ITAU_P', 2, 40)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'DEGA_IGAIN_P', 5, 80)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'DESC_ITAU_P', 3, 20)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'DESC_IGAIN_P', 4, 80)
    set_parameter(myConfig.chips[0].cores[1].parameters, 'DESC_ITAU_P', 3, 20)
    set_parameter(myConfig.chips[0].cores[1].parameters, 'DESC_IGAIN_P', 4, 80)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'SYAM_W0_P', 4, 160)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'SYAM_W1_P', 5, 160)
    set_parameter(myConfig.chips[0].cores[1].parameters, 'SYAM_W1_P', 5, 160)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'SYAM_W2_P', 5, 80)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'SYAM_W3_P', 5, 80)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'SYPD_EXT_N', 3, 200)
    set_parameter(myConfig.chips[0].cores[1].parameters, 'SYPD_EXT_N', 3, 200)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'SYPD_DLY0_P', 1, 60)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'SYPD_DLY1_P', 5, 200)
    set_parameter(myConfig.chips[0].cores[0].parameters, 'SYPD_DLY2_P', 1, 20)


def hcm(board, number_of_chips, profile_path):

    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("Configuring paramrters")
    config_parameters(myConfig=myConfig)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    for core in range(4):
        for neuron in range(256):
            synapses = [Dynapse2Synapse() for _ in range(64)]
            for synapse in synapses:
                synapse.tag = random.randint(0, 2047)
            myConfig.chips[0].cores[core].neurons[neuron].synapses = synapses

    network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips)
    dummy_input = network.add_virtual_group(size=1)
    input_A = network.add_virtual_group(size=1)
    input_B = network.add_virtual_group(size=1)
    neuron_A = network.add_group(chip=0, core=0, size=1, neurons=[0])
    neuron_A_plus = network.add_group(chip=0, core=0, size=1, neurons=[1])
    neuron_A_plus_plus = network.add_group(chip=0, core=0, size=1, neurons=[2])
    neuron_B = network.add_group(chip=0, core=0, size=1, neurons=[3])
    neuron_B_plus = network.add_group(chip=0, core=0, size=1, neurons=[4])
    neuron_B_inh = network.add_group(chip=0, core=1, size=1, neurons=[5])
    neuron_AB = network.add_group(chip=0, core=0, size=1, neurons=[6])
    dummy_neuron = network.add_group(chip=0, core=0, size=1)

    network.add_connection(source=dummy_input, target=dummy_neuron, probability=1,
                           dendrite=Dendrite.none, weight=[False, False, False, False], repeat=16)
    network.add_connection(source=input_A, target=neuron_A, probability=1,
                           dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=16, precise_delay=True)
    network.add_connection(source=neuron_A, target=neuron_A_plus, probability=1,
                           dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=16)
    network.add_connection(source=neuron_A_plus, target=neuron_A_plus_plus, probability=1,
                           dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=16)
    network.add_connection(source=input_B, target=neuron_B, probability=1,
                           dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=16, precise_delay=True)
    network.add_connection(source=neuron_B, target=neuron_B_plus, probability=1,
                           dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=16)
    network.add_connection(source=neuron_B, target=neuron_B_inh, probability=1,
                           dendrite=Dendrite.shunt, weight=[False, True, False, False], repeat=16)
    network.add_connection(source=neuron_A_plus, target=neuron_AB, probability=1,
                           dendrite=Dendrite.nmda, weight=[False, False, True, False], repeat=16, precise_delay=True)
    network.add_connection(source=neuron_B_inh, target=neuron_AB, probability=1,
                           dendrite=Dendrite.shunt, weight=[False, True, False, False], repeat=16, precise_delay=True)
    network.add_connection(source=neuron_AB, target=neuron_A_plus_plus, probability=1,
                           dendrite=Dendrite.gaba, weight=[False, False, False, True], repeat=16, mismatched_delay=True)
    network.add_connection(source=neuron_AB, target=neuron_B_plus, probability=1,
                           dendrite=Dendrite.gaba, weight=[False, False, False, True], repeat=16, mismatched_delay=True)
    network.connect()
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    myConfig.chips[0].cores[0].monitored_neuron = neuron_AB.neurons[0]
    myConfig.chips[0].cores[0].neuron_monitoring_on = True
    myConfig.chips[0].cores[1].neurons[neuron_B_inh.neurons[0]].latch_so_dc = True
    myConfig.chips[0].cores[1].monitored_neuron = neuron_B_inh.neurons[0]
    myConfig.chips[0].cores[1].neuron_monitoring_on = True
    model.apply_configuration(myConfig)
    # time.sleep(0.1)

    for _ in input_A.get_destinations().values():
        tag_A = _[0]
    for _ in input_B.get_destinations().values():
        tag_B = _[0]

    print("\nAll configurations done!\n")

    ts = get_fpga_time(board=board) + 10000

    tag_list = [tag_A, tag_A, tag_A, tag_A, tag_B, tag_A, tag_A, tag_B,
                tag_B, tag_A, tag_B, tag_A, tag_B, tag_B, tag_B, tag_B]
    input_events = []
    for tag in tag_list:
        input_events += VirtualSpikeConstructor(tag=tag, core=[True, False, False, False],timestamp=ts).spikes
        ts += 10000
    # print(input_events)
    send_events(board=board, events=input_events, min_delay=0)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    # spike_count(output_events=output_events)
    plot_raster(output_events=output_events)

