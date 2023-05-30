import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_spikegen import get_fpga_time, send_events
from lib.dynapse2_raster import get_events, plot_raster, spike_count
from lib.dynapse2_obj import *
from samna.dynapse2 import *
from multiprocessing import Process, Manager, Lock, Array

import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

tau = 0.0001


def print_weight(tags):
    w = [np.bincount([tags[i][_] - 1024 for _ in range(48)], minlength=16) for i in range(16)]
    for i in range(16):
        print(str(w[i]))


def interact(l1, l2, l3, n, board, model, myConfig, tags, output_aer, output_ts, all_aer, all_ts):
    l1.acquire()
    l3.acquire()
    time.sleep(0.1)
    synapses = [[Dynapse2Synapse() for _ in range(64)] for i in range(16)]
    for i in range(16):
        for j in range(48):
            synapses[i][j].weight = [True, False, False, False]
            synapses[i][j].dendrite = Dendrite.ampa
            tags[i][j] = 1024 + random.randint(0, 15)
        for j in range(48, 64):
            synapses[i][j].weight = [False, True, False, False]
            synapses[i][j].tag = 1024 + 255
            synapses[i][j].dendrite = Dendrite.shunt
    tags_buf = [[tags[i][j] for j in range(64)] for i in range(16)]
    for _ in range(n):
        if _ % 10 == 0:
            for i in range(16):
                for j in range(48):
                    synapses[i][j].tag = tags_buf[i][j]
                myConfig.chips[0].cores[0].neurons[i].synapses = synapses[i]
            model.apply_configuration(myConfig)
            print("Iter " + str(_))
            print_weight(tags=tags_buf)
        ts = get_fpga_time(board=board) + 100000
        input_events = []
        for x in range(16):
            for t in range(ts + x * 10000, ts + (x + 1) * 10000, 100):
                input_events += VirtualSpikeConstructor(1024 + int(x), [True, False, False, False], t).spikes
        send_events(board=board, events=input_events)
        received_last_input = False
        while not received_last_input:
            for ev in board.read_events():
                tag = ev.event.tag
                if ev.event.y_hop == 0:
                    neuron_id = tag + (ev.event.x_hop + 6) * 2048
                    t = ev.timestamp * 1e-6
                    output_aer += [neuron_id]
                    output_ts += [t]
                elif tag == 2047:
                    received_last_input = True
                    break
        all_aer += output_aer[:]
        all_ts += output_ts[:]
        l1.release()
        l2.acquire()
        for i in range(16):
            for j in range(48):
                tags_buf[i][j] = tags[i][j]
        # print_weight(tags=tags)
        l3.release()
        l1.acquire()
        l2.release()
        l3.acquire()
    l1.release()
    l3.release()


def learning(l1, l2, l3, n, tags, output_aer, output_ts):
    a_pre = np.zeros(17)
    t_last = 0
    l2.acquire()
    time.sleep(0.1)
    for _ in range(n):
        l1.acquire()
        # print(len(output_aer))
        output_events = [output_aer[:], output_ts[:]]
        output_aer[:] = []
        output_ts[:] = []
        l2.release()
        l3.acquire()
        l1.release()
        l2.acquire()
        l3.release()
        # print(len(output_events[0]))
        for i in range(len(output_events[0])):
            neuron_id = output_events[0][i]
            t = output_events[1][i]
            # print(neuron_id)
            if neuron_id < 1024:
                if random.random() < 0.005:
                    for pre in range(16):
                        if a_pre[pre] > a_pre[16]:
                            tags[neuron_id][random.randint(0, 47)] = 1024 + pre
            else:
                a_pre[neuron_id - 1024] += 1
                a_pre[16] += 1 / 16
            if t > t_last:
                a_pre *= np.exp(-(t - t_last) / tau)
                t_last = t
    l2.release()


def corr(output_events):
    count = [[0 for i in range(16)] for j in range(16)]
    for i in range(len(output_events[0])):
        neuron_id = output_events[0][i]
        if neuron_id < 16:
            count[neuron_id][int((output_events[1][i] - output_events[1][0])/0.01) % 16] += 1
    for i in range(16):
        print(count[i])


def config_parameters(myConfig):
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOAD_PWTAU_N", 4, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOCA_W_N", 2, 100)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOCA_GAIN_P", 1, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOCA_TAU_P", 0, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VB_P", 5, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VH_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_L_P", 1, 30)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_M_P", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_H_P", 3, 120)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VRST_P", 4, 80)


def homeostasis_time_constant(myConfig):
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_L_P", 2, 30)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_M_P", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_H_P", 2, 120)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VRST_P", 4, 80)


def homeostasis_reference(myConfig):
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_P", 1, 40)


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

    print("Homeostasis enable")
    homeostasis_enable(myConfig, range(16))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 1, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_ITAU_P', 1, 30)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 3, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 3, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 160)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    config_parameters(myConfig)
    model.apply_configuration(myConfig)
    time.sleep(1)

    # set CAM -- synapses
    synapses = [[[Dynapse2Synapse() for _ in range(64)] for i in range(256)] for c in range(4)]
    for c in range(1):
        for i in range(16):
            for j in range(48):
                synapses[c][i][j].weight = [True, False, False, False]
                synapses[c][i][j].tag = 1024 + random.randint(0, 15)
                synapses[c][i][j].dendrite = Dendrite.ampa
            for j in range(48, 64):
                synapses[c][i][j].weight = [False, True, False, False]
                synapses[c][i][j].tag = 1024 + 255
                synapses[c][i][j].dendrite = Dendrite.shunt
            myConfig.chips[0].cores[c].neurons[i].synapses = synapses[c][i]
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set SRAM -- axons
    for c in range(1):
        for n in range(16):
            myConfig.chips[0].cores[c].neurons[n].destinations = \
                [DestinationConstructor(tag=c * 256 + n, core=[True] * 4, x_hop=-7).destination,
                 DestinationConstructor(tag=c * 256 + n, core=[True] * 4, x_hop=0).destination,
                 DestinationConstructor(tag=(c + 1) * 256 + 1024 - 1, core=[True] * 4, x_hop=0).destination,
                 Dynapse2Destination()]
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("Homeostasis activate")
    homeostasis_activate(myConfig, range(16))
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

    tags = [Array('i', 64) for _ in range(16)]
    l1 = Lock()
    l2 = Lock()
    l3 = Lock()
    n = 101
    with Manager() as manager:
        output_aer = manager.list()
        output_ts = manager.list()
        all_aer = manager.list()
        all_ts = manager.list()

        p2 = Process(target=learning, args=(l1, l2, l3, n, tags, output_aer, output_ts))
        # p1.start()
        p2.start()
        interact(l1, l2, l3, n, board, model, myConfig, tags, output_aer, output_ts, all_aer, all_ts)
        # p1.join()
        p2.join()

        plot_raster(output_events=[all_aer[:], all_ts[:]])

