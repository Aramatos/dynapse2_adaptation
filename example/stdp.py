import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_spikegen import get_fpga_time, send_events
from lib.dynapse2_raster import get_events, plot_raster, spike_count
from lib.dynapse2_obj import *
from samna.dynapse2 import *

import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

dt = 10
n_step = 1000
tau = 0.0001
n_neuron = 16
n_dim = 2
n_length = 4
n_pattern = n_length ** n_dim
n_input = n_length * n_dim
preQueue = []


def print_weight(synapses):
    w = [np.bincount([synapses[i][_].tag - 1024 for _ in range(48)], minlength=n_input) for i in range(n_neuron)]
    for i in range(n_neuron):
        print("[", end="")
        for _ in w[i]:
            print('{:2d}'.format(_), end=",")
        print("],")


def learning(board, synapses, output_events, ts):
    # a_pre = np.zeros(n_input + 1)
    t_last = 0

    received_last_input = False
    while not received_last_input:
        for ev in board.read_events():
            tag = ev.event.tag
            if ev.event.y_hop == 0:
                neuron_id = tag + (ev.event.x_hop + 6) * 2048
                t = ev.timestamp * 1e-6
                output_events[0] += [neuron_id]
                output_events[1] += [t]
                if neuron_id < 1024:
                    if random.random() < 0.1:
                        # for pre in range(n_input):
                        #     if a_pre[pre] > a_pre[n_input]:
                        #         synapses[neuron_id][random.randint(0, 47)].tag = 1024 + pre
                        # print(preQueue)
                        synapses[neuron_id][random.randint(0, 47)].tag = preQueue[random.randint(0, len(preQueue)-1)]
                else:
                    preQueue.append(neuron_id)
                    if len(preQueue) > 16:
                        preQueue.pop(0)
                    # a_pre[neuron_id - 1024] += 1
                    # a_pre[n_input] += 1 / n_input
                # if t > t_last:
                    # a_pre *= np.exp(-(t - t_last) / tau)
                    # t_last = t
            else:
                # print(ev.timestamp)
                # print(ts)
                if tag == 2047 and ev.timestamp > ts:
                    received_last_input = True
                break


def config_parameters(myConfig):
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOAD_PWTAU_N", 3, 160)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOCA_W_N", 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOCA_GAIN_P", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOCA_TAU_P", 0, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VB_P", 5, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VH_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_P", 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_L_P", 1, 30)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_M_P", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_H_P", 3, 120)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VRST_P", 2, 40)


def homeostasis_time_constant(myConfig):
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_L_P", 2, 15)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_M_P", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_H_P", 2, 240)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VRST_P", 4, 80)


def homeostasis_reference(myConfig):
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_P", 2, 40)


def homeostasis_enable(myConfig, neurons):
    # set neuron latches
    for c in range(1):
        for n in neurons:
            myConfig.chips[0].cores[c].neurons[n].latch_ho_enable = True


def homeostasis_disable(myConfig, neurons):
    # set neuron latches
    for c in range(1):
        for n in neurons:
            myConfig.chips[0].cores[c].neurons[n].latch_ho_enable = False


def homeostasis_activate(myConfig, neurons):
    # set neuron latches
    for c in range(1):
        for n in neurons:
            myConfig.chips[0].cores[c].neurons[n].latch_ho_active = True


def learn_to_divide(board, number_of_chips):



    # your code starts here
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    # set initial configuration
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    neuron = 5
    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set dc latches
    set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("Homeostasis enable")
    homeostasis_enable(myConfig, range(n_neuron))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 0, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 4, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 4, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_ITAU_P', 4, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', 4, 160)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 4, 100)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 5, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 80)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    config_parameters(myConfig)
    model.apply_configuration(myConfig)
    time.sleep(1)

    # set CAM -- synapses
    synapses = [[Dynapse2Synapse() for _ in range(64)] for i in range(n_neuron)]
    for i in range(n_neuron):
        for j in range(48):
            synapses[i][j].weight = [True, False, False, False]
            synapses[i][j].tag = 1024 + random.randint(0, n_input - 1)
            synapses[i][j].dendrite = Dendrite.ampa
        for j in range(48, 64):
            synapses[i][j].weight = [False, True, False, False]
            synapses[i][j].tag = 1024 + 255
            synapses[i][j].dendrite = Dendrite.shunt
        myConfig.chips[0].cores[0].neurons[i].synapses = synapses[i]
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set SRAM -- axons
    for c in range(1):
        for n in range(n_neuron):
            myConfig.chips[0].cores[c].neurons[n].destinations = \
                [DestinationConstructor(tag=c * 256 + n, core=[True] * 4, x_hop=-7).destination,
                 DestinationConstructor(tag=c * 256 + n, core=[True] * 4, x_hop=0).destination,
                 DestinationConstructor(tag=(c + 1) * 256 + 1024 - 1, core=[True] * 4, x_hop=0).destination,
                 Dynapse2Destination()]
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("Homeostasis activate")
    homeostasis_activate(myConfig, range(n_neuron))
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("Homeostasis reset")
    homeostasis_time_constant(myConfig)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("Homeostasis start")
    homeostasis_reference(myConfig)
    model.apply_configuration(myConfig)
    time.sleep(1)

    all_events = [[], []]
    x = [i for i in range(n_pattern)]
    random.shuffle(x)
    patterns = [[x[j] % n_length, int(x[j]/n_length) + n_length] for j in range(n_pattern)]

    for _ in range(801):
        ts = get_fpga_time(board=board) + n_pattern * 5000
        t_last = ts
        # print(ts)
        input_events = []
        for x in range(n_pattern):
            for i in range(n_step):
                if random.random() < 0.1:
                    # d = (i / n_step - 0.5)
                    d = 0
                    j = random.choices(population=[(x - 1) % n_pattern, x, (x + 1) % n_pattern],
                                       weights=[max(-d, 0), 1 - abs(d), max(d, 0)])[0]
                    input_events += VirtualSpikeConstructor(1024 + random.choice(patterns[j]),
                                                            [True, False, False, False], ts).spikes
                    t_last = ts
                ts += dt
        send_events(board=board, events=input_events)
        output_events = [[], []]
        learning(board=board, synapses=synapses, output_events=output_events, ts=t_last)
        all_events[0] += output_events[0]
        all_events[1] += output_events[1]
        if _ % 10 == 0:
            # print("Iter " + str(_))
            for i in range(n_neuron):
                myConfig.chips[0].cores[0].neurons[i].synapses = synapses[i]
            model.apply_configuration(myConfig)
            if _ % 100 == 0:
                print_weight(synapses=synapses)
                print("],[")
            # corr(output_events=output_events)
    plot_raster(output_events=all_events)

