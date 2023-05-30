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


wta_core = 0


def config_parameters(myConfig, delay, stp):
    # set neuron parameters
    set_parameter(myConfig.chips[0].shared_parameters01, "PG_BUF_N", 0, 100)
    set_parameter(myConfig.chips[0].shared_parameters23, "PG_BUF_N", 0, 100)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, "SOIF_GAIN_N", 3, 80)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, "SOIF_LEAK_N", 0, 50)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, "SOIF_REFR_N", 1, 40)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, "SOIF_DC_P", 0, 1)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, "SOIF_SPKTHR_P", 3, 254)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'DEAM_ETAU_P', 2, 40)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'DEAM_EGAIN_P', 4, 80)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'DENM_ETAU_P', 1, 40)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'DENM_EGAIN_P', 1, 160)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'DESC_ITAU_P', 1, 40)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'DESC_IGAIN_P', 4, 80)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'SYAM_W0_P', 4, 80)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'SYAM_W1_P', 5, 160)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'SYAM_W2_P', 5, 160)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'SYAM_W3_P', 5, 80)
    set_parameter(myConfig.chips[0].cores[wta_core].parameters, 'SYPD_EXT_N', 3, 200)



def config_connections(network, virtual_group_size, num_groups):

    input_groups = [network.add_virtual_group(size=virtual_group_size) for _ in range(num_groups)]
    exc_groups = [[network.add_group(chip=0, core=wta_core, size=4, neurons=[0,2,3]) for _ in range(2)] for _ in range(num_groups)]
    for input_group, group in zip(input_groups, exc_groups):
        network.add_connection(source=input_group, target=group[0], probability=0.25,
                               dendrite=Dendrite.ampa, weight=[True, False, False, False], repeat=1)
        network.add_connection(source=group[0], target=group[1], probability=1,
                               dendrite=Dendrite.ampa, weight=[False, True, False, False], repeat=1)
        network.add_connection(source=group[1], target=group[1], probability=1,
                               dendrite=Dendrite.nmda, weight=[False, False, True, False], repeat=1)
        # network.add_connection(source=group, target=inh_group, probability=0.25,
        #                        dendrite=Dendrite.ampa, weight=[False, False, True, False], repeat=1)
        for other_group in exc_groups:
            if other_group != group:
                network.add_connection(source=group[1], target=other_group[1], probability=1,
                                       dendrite=Dendrite.shunt, weight=[False, False, False, True], repeat=1)

    network.connect()
    for group in network.groups:
        print(group.neurons)


def wta_basic(board, number_of_chips, profile_path, delay=False, adaptation=False, stp=False):

    # your code starts here
    input_group_size = 16
    num_groups = 4

    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("Configuring paramrters")
    config_parameters(myConfig=myConfig, delay=delay, stp=stp)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips)
    config_connections(network=network, virtual_group_size=input_group_size, num_groups=num_groups)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    myConfig.chips[0].cores[wta_core].monitored_neuron = network.groups[1].neurons[0]
    myConfig.chips[0].cores[wta_core].neuron_monitoring_on = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("\nAll configurations done!\n")

    input_events = [event for t, i in enumerate(list(range(num_groups)) + list(range(num_groups - 2, -1, -1)))
                    for event in poisson_gen(
            start=t * 1000000,
            duration=500000,
            virtual_groups=network.virtual_groups,
            rates=[100 * (j == i) for j in range(num_groups)])]
    ts = get_fpga_time(board=board) + 1000000
    send_virtual_events(board=board, virtual_events=input_events, offset=ts, min_delay=500000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)

