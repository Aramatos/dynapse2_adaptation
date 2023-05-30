import time
import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import set_parameter, clear_srams
from lib.dynapse2_spikegen import get_fpga_time, send_events
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *

import random

from samna.dynapse2 import *


def config_latches(myConfig, neurons):
    # for each core, set the neuron to monitor
    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neurons[-1]
        for n in neurons:
            myConfig.chips[0].cores[c].neurons[n].latch_ho_enable = True


def config_parameters(myConfig):
    # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 0, 50)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 50)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_CC_N", 5, 254)

    # set synapse parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 4, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 4, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', 3, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 80)

    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOAD_PWTAU_N", 4, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOCA_W_N", 4, 100)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOCA_GAIN_P", 0, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOCA_TAU_P", 0, 100)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VB_P", 5, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VH_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_L_P", 1, 30)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_M_P", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_H_P", 3, 120)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VRST_P", 4, 80)


def homeostasis_time_constant(myConfig):
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_L_P", 2, 15)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_M_P", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_H_P", 2, 240)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VRST_P", 4, 80)


def homeostasis_reference(myConfig):
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOHO_VREF_P", 1, 100)


def homeostasis_activate(myConfig, neurons):
    # set neuron latches
    for c in range(1):
        for n in neurons:
            myConfig.chips[0].cores[c].neurons[n].latch_ho_active = True


def config_cams(myConfig, neurons):
    # set CAM -- synapses
    for c in range(1):
        for i in range(256):
            cams = [Dynapse2Synapse() for _ in range(64)]
            for j in range(64):
                weights = [False, False, False, False]
                if i in neurons:
                    weights[0] = True
                    cams[j].tag = random.randint(i - 3, i + 3) + 1024
                    cams[j].dendrite = Dendrite.ampa
                else:
                    weights[0] = True
                    cams[j].tag = 0
                    cams[j].dendrite = Dendrite.none
                cams[j].weight = weights
            myConfig.chips[0].cores[c].neurons[i].synapses = cams


def generate_events(board, neurons, input_events):

    ts = get_fpga_time(board=board) + 100000

    for t in range(ts, ts + 10000000, 100):
        k = 1024 + random.choices(neurons, [_ for _ in range(len(neurons))])[0] + random.randint(-3, 3)
        if random.random() < 0.01:
            input_events += VirtualSpikeConstructor(k, [True, False, False, False], t).spikes


def homeostasis(board, number_of_chips):

    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    neurons = range(10, 20)

    print("Configuring latches")
    config_latches(myConfig, neurons)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("Configuring paramrters")
    config_parameters(myConfig)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("configuring cams")
    config_cams(myConfig, neurons)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("configuring srams")
    clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips), all_to_all=True)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("\nAll configurations done!\n")

    print("Homeostasis activate")
    homeostasis_activate(myConfig, neurons)
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

    input_events = []
    generate_events(board, neurons, input_events)
    send_events(board=board, events=input_events, min_delay=100000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)
    
    
    
def homeostasis_sadc(board, number_of_chips):

    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    neurons = range(10, 20)

    print("Configuring latches")
    config_latches(myConfig, neurons)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("Configuring paramrters")
    config_parameters(myConfig)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("configuring cams")
    config_cams(myConfig, neurons)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("configuring srams")
    clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips), all_to_all=True)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("\nAll configurations done!\n")

    print("Homeostasis activate")
    homeostasis_activate(myConfig, neurons)
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

    input_events = []
    generate_events(board, neurons, input_events)

    set_parameter(myConfig.chips[0].shared_parameters01, "NCCF_CAL_OFFBIAS_P", 1, 255)
    set_parameter(myConfig.chips[0].shared_parameters23, "NCCF_CAL_OFFBIAS_P", 1, 255)
    for core in myConfig.chips[0].cores:
        core.sadc_enables.soif_mem = False
        core.sadc_enables.soif_refractory = True
        core.sadc_enables.soad_dpi = False
        core.sadc_enables.soca_dpi = False
        core.sadc_enables.deam_edpi = True
        core.sadc_enables.deam_idpi = True
        core.sadc_enables.denm_edpi = True
        core.sadc_enables.denm_idpi = True
        core.sadc_enables.dega_idpi = True
        core.sadc_enables.desc_idpi = True
        core.sadc_enables.sy_w42 = True
        core.sadc_enables.sy_w21 = True
        core.sadc_enables.soho_sogain = True
        core.sadc_enables.soho_degain = True
    myConfig.chips[0].sadc_enables.nccf_extin_vi_group0_pg1 = True
    myConfig.chips[0].sadc_enables.nccf_cal_refbias_v_group1_pg1 = True
    myConfig.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg1 = True
    myConfig.chips[0].sadc_enables.nccf_extin_vi_group2_pg1 = True
    myConfig.chips[0].sadc_enables.nccf_extin_vi_group0_pg0 = True
    myConfig.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg0 = True
    myConfig.chips[0].sadc_enables.nccf_extin_vi_group2_pg0 = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    for i in range(3):
        set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_PWLK_P", 5, 160)
        set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_PWLK_P", 5, 160)
        set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_HYS_P", 0, 0)
        set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_HYS_P", 0, 0)
        set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_BIAS_P", 5, 200)
        set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_BIAS_P", 5, 200)
        set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_REF_L_V", 0, 100)
        set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_REF_L_V", 0, 100)
        set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_REF_H_V", 3, 80)
        set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_REF_H_V", 3, 80)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    model.set_sadc_sample_period_ms(10)
    board.enable_output(BusId.sADC, True)
    board.enable_output(BusId.W, True)

    send_events(board=board, events=input_events, min_delay=100000)

    while True:
        sadcValues = model.get_sadc_values(0)
        time.sleep(0.01)
        # for _ in range(300):
        #     time.sleep(0.00001)
            # for ev in board.read_events():
            #     if ev.event.tag < 1024:
            #         print(ev.event.tag)
                # pass
        # print(f"{get_sadc_description(42)}: {sadcValues[42]}; {get_sadc_description(57)}: {sadcValues[57]}")
        print(f"{sadcValues[40]}, {sadcValues[59]}")

