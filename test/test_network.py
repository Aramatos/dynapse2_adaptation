import time
import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_spikegen import send_events
from lib.dynapse2_raster import *
from samna.dynapse2 import *


def network_full(board, number_of_chips):

    model = board.get_model()
    model.reset(ResetType.ConfigReset, (1 << number_of_chips) - 1)
    time.sleep(1)

    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    neurons = range(16)

    # set neuron parameters
    print("Setting parameters PC")
    h=0
    c=0
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_GAIN_N", 2, 40)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_LEAK_N", 0, 1)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_REFR_N", 3, 255)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_DC_P", 3, 80)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_SPKTHR_P", 4, 255)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("Setting parameters PV")
    h=0
    c=1
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_GAIN_N", 2, 40)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_LEAK_N", 0, 1)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_REFR_N", 3, 255)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_DC_P", 3, 80)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_SPKTHR_P", 4, 255)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("Setting parameters SST")
    h=0
    c=2
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_GAIN_N", 2, 40)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_LEAK_N", 0, 1)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_REFR_N", 3, 255)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_DC_P", 3, 80)
    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_SPKTHR_P", 4, 255)
    model.apply_configuration(myConfig)
    time.sleep(1)

    # set neurons to monitor
    print("Setting monitors")
    for h in range(number_of_chips):
        for c in range(3):
            myConfig.chips[h].cores[c].neuron_monitoring_on = True
            myConfig.chips[h].cores[c].monitored_neuron = 10  # monitor neuron 3 on each core
            myConfig.chips[h].cores[c].enable_pulse_extender_monitor1 = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set neuron latches to get DC input
    print("Setting DC")
    set_dc_latches(config=myConfig, neurons=neurons, cores=range(1), chips=range(number_of_chips))
    # set_type_latches(config=myConfig, neurons=neurons, cores=range(1), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set SRAM -- axons
    print("Setting SRAMs")
    clear_srams(config=myConfig, neurons=neurons, cores=range(1), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("\nAll configurations done!\n")
    send_events(board=board, events=[], min_delay=10000000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)
