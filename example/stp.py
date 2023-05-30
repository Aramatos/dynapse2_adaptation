import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_spikegen import get_fpga_time, send_events
from lib.dynapse2_raster import get_events, plot_raster, spike_count
from lib.dynapse2_obj import *
from samna.dynapse2 import *


def short_term_potentiation(board, number_of_chips):

    # your code starts here
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    # set initial configuration
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    neuron = 0x00
    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set dc latches
    set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 0, 30)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 1, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 3, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 160)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    synapses = [[[Dynapse2Synapse() for _ in range(64)] for i in range(256)] for c in range(4)]
    for c in range(1):
        for i in range(256):
            for j in range(64):
                weights = [True, False, False, False]
                synapses[c][i][j].weight = weights
                if 0 <= j < 2:
                    synapses[c][i][j].tag = 1024 + i
                    synapses[c][i][j].dendrite = Dendrite.ampa
                else:
                    synapses[c][i][j].tag = 0
                    synapses[c][i][j].dendrite = Dendrite.none
            myConfig.chips[0].cores[c].neurons[i].synapses = synapses[c][i]
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set SRAM -- axons
    clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    ts = get_fpga_time(board=board) + 5000000

    input_events = []
    for j in [6, 5, 4, 3, 2, 1]:
        for i in range(256):
            for t in range(ts, ts+60, j):
                input_events += VirtualSpikeConstructor(1024+i, [True, False, False, False], t).spikes
            ts += 10000

    print("\nAll configurations done!\n")
    send_events(board=board, events=input_events, min_delay=100000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)
