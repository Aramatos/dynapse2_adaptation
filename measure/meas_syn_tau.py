import itertools
import time
import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import set_parameter, clear_srams
from lib.dynapse2_obj import *
from lib.dynapse2_spikegen import get_fpga_time, send_events
from lib.dynapse2_raster import *

from samna.dynapse2 import *

N_neurons = 256
N_cores = 1


def set_neuron_to_monitor(model, myConfig, neuron):
    # for each core, set the neuron to monitor
    # neuron = 2
    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
    model.apply_configuration(myConfig)
    time.sleep(0.1)

def common(model, myConfig, number_of_chips, dendrite):

    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    # set initial configuration
    model.apply_configuration(myConfig)
    time.sleep(1)

    set_neuron_to_monitor(model=model, myConfig=myConfig, neuron=0)

    # # set neuron parameters
    for c in range(N_cores):
        if dendrite == Dendrite.ampa or dendrite == Dendrite.nmda:
            set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 254)
        elif dendrite == Dendrite.gaba or dendrite==Dendrite.shunt:
            set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 5, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_CC_N", 4, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set synapse parameters  -- enabled AM and SC
    for c in range(N_cores):
        if dendrite == Dendrite.ampa:
            set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 1, 140)
            set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 1, 20)
        elif dendrite == Dendrite.nmda:
            set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 1, 140)
            set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 1, 20)
        elif dendrite == Dendrite.gaba:
            set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_ITAU_P', 1, 140)
            set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_IGAIN_P', 4, 20)
        elif dendrite == Dendrite.shunt:
            set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_ITAU_P', 1, 140)
            set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', 1, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 5, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 4, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', 4, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', 4, 120)
        if dendrite == Dendrite.ampa or dendrite == Dendrite.nmda or dendrite == Dendrite.shunt:
            set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 4, 80)
        elif dendrite == Dendrite.gaba:
            set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 4, 170)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(N_cores):
        for i in range(N_neurons):
            cam_exc = [Dynapse2Synapse() for _ in range(64)]
            for j in range(64):
                weights = [False, False, False, False]
                weights[j % 4] = True
                cam_exc[j].weight = weights
                if j < 1:
                    cam_exc[j].tag = c * 256 + i + 1024
                    cam_exc[j].dendrite = Dendrite.ampa
                else:
                    cam_exc[j].tag = 0
                    cam_exc[j].dendrite = Dendrite.none
            myConfig.chips[0].cores[c].neurons[i].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set SRAM -- axons
    clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)


def sweep(board, model, myConfig, number_of_chips, params):

    for f in itertools.product(*[params[_][2] for _ in range(len(params))]):
        print(f)
        for i in range(len(params)):
            print(params[i][0] + " = (" + str(params[i][1]) + ", " + str(f[i]) + ")")
            for c in range(N_cores):
                set_parameter(myConfig.chips[0].cores[c].parameters, params[i][0], params[i][1], f[i])

        model.apply_configuration(myConfig)
        time.sleep(0.1)

        ts = get_fpga_time(board=board) + 10000000

        input_events = []
        for c in range(N_cores):
            for i in range(N_neurons):
                input_events += VirtualSpikeConstructor((c * 256 + i + 1024), [True, False, False, False],
                                                        ts).spikes
                ts += 10000

        send_events(board=board, events=input_events, min_delay=0)
        output_events = [[], []]
        get_events(board=board, extra_time=100, output_events=output_events)


def syn_tau_ampa(board, number_of_chips):
    model = board.get_model()
    myConfig = model.get_configuration()
    common(model=model, myConfig=myConfig, number_of_chips=number_of_chips, dendrite=Dendrite.ampa)
    # sweep(board=board, model=model, myConfig=myConfig, number_of_chips=number_of_chips,
    #       params=[("SOIF_LEAK_N", 1, [50, 100, 150, 200, 250]),
    #               ("DEAM_ETAU_P", 1, [240, 120, 80, 60, 48])])
    for neuron in range(N_neurons):
        set_neuron_to_monitor(model=model, myConfig=myConfig, neuron=neuron)
        sweep(board=board, model=model, myConfig=myConfig, number_of_chips=number_of_chips,
              params=[("SOIF_LEAK_N", 0, [0]),
                      ("DEAM_ETAU_P", 1, [120])])
        time.sleep(10)


def syn_tau_gaba(board, number_of_chips):
    model = board.get_model()
    myConfig = model.get_configuration()
    common(model=model, myConfig=myConfig, number_of_chips=number_of_chips, dendrite=Dendrite.gaba)
    sweep(board=board, model=model, myConfig=myConfig, number_of_chips=number_of_chips,
          params=[("SOIF_DC_P", 2, [30, 35, 40, 45, 50]),
                  ("DEGA_ITAU_P", 1, [240, 120, 80, 60, 48])])

def syn_tau_shunt(board, number_of_chips):
    model = board.get_model()
    myConfig = model.get_configuration()
    common(model=model, myConfig=myConfig, number_of_chips=number_of_chips, dendrite=Dendrite.gaba)
    sweep(board=board, model=model, myConfig=myConfig, number_of_chips=number_of_chips,
          params=[("SOIF_DC_P", 2, [30, 35, 40, 45, 50]),
                  ("DESC_ITAU_P", 1, [240, 120, 80, 60, 48])])
