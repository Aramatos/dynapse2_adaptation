import time
import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_spikegen import send_events, get_fpga_time

from lib.dynapse2_raster import *
from samna.dynapse2 import *

def test_calibration(board, number_of_chips):

    model = board.get_model()
    model.reset(ResetType.ConfigReset, 0b1)

    # config = model.get_configuration()
    # config.chips[0].cores[0].monitored_neuron = 1
    # set_parameter(config.chips[0].global_parameters, 'R2R_BUFFER_AMPB', 5, 255)
    # set_parameter(config.chips[0].global_parameters, 'R2R_BUFFER_CCB', 5, 255)
    # set_parameter(config.chips[0].shared_parameters01, 'LBWR_VB_P', 3, 255)
    # set_parameter(config.chips[0].cores[0].parameters, 'SOIF_GAIN_N', 2, 255)
    # set_parameter(config.chips[0].cores[0].parameters, 'SOIF_LEAK_N', 0, 0)
    # set_parameter(config.chips[0].cores[0].parameters, 'SOIF_REFR_N', 4, 255)
    # set_parameter(config.chips[0].cores[0].parameters, 'SOIF_DC_P', 1, 255)
    # set_parameter(config.chips[0].cores[0].parameters, 'SOIF_SPKTHR_P', 3, 255)
    # set_parameter(config.chips[0].cores[0].parameters, 'SOIF_CC_N', 3, 255)
    # config.chips[0].cores[0].neuron_monitoring_on = True
    # config.chips[0].cores[0].neurons[1].latch_so_dc = True
    #
    # config.chips[0].sadc_enables.nccf_cal_refbias_v_group1_pg0 = True
    #
    # # Disable all other sADC channels - this is optional
    # for core in config.chips[0].cores:
    #     core.sadc_enables.soif_mem = False
    #     core.sadc_enables.soif_refractory = False
    #     core.sadc_enables.soad_dpi = False
    #     core.sadc_enables.soca_dpi = False
    #     core.sadc_enables.deam_edpi = False
    #     core.sadc_enables.deam_idpi = False
    #     core.sadc_enables.denm_edpi = False
    #     core.sadc_enables.denm_idpi = False
    #     core.sadc_enables.dega_idpi = False
    #     core.sadc_enables.desc_idpi = False
    #     core.sadc_enables.sy_w42 = False
    #     core.sadc_enables.sy_w21 = False
    #     core.sadc_enables.soho_sogain = False
    #     core.sadc_enables.soho_degain = False
    # config.chips[0].sadc_enables.nccf_extin_vi_group0_pg1 = False
    # config.chips[0].sadc_enables.nccf_cal_refbias_v_group1_pg1 = False
    # config.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg1 = False
    # config.chips[0].sadc_enables.nccf_extin_vi_group2_pg1 = False
    # config.chips[0].sadc_enables.nccf_extin_vi_group0_pg0 = False
    # config.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg0 = False
    # config.chips[0].sadc_enables.nccf_extin_vi_group2_pg0 = False
    #
    # # Get address info for sADC for NCCF_CAL_REFBIAS_V, group 1, parameter
    # # generator 0.
    # # The address is necessary when calling get_sadc_value() in the loop
    # # below. Note that if you want to get the address of one of the cores'
    # # sADCs, the name of which doesn't start with NCCF, then you need to use
    # # sadc_lookup_aer_address(), e.g. the address for core 2's
    # # SOIF_MEM is found by calling sadc_lookup_aer_address('SOIF_MEM', 2).
    #
    # nccf_cal_ref_g1_pg0_address = sadc_lookup_aer_address_nccf(
    #     'NCCF_CAL_REFBIAS_V', 1, ParamGen.PG0)
    #
    # model.apply_configuration(config)
    # time.sleep(1)
    #
    # board.enable_output(BusId.sADC, True)
    #
    # try:
    #     while True:
    #         for ev in board.output_read():
    #             print('%08x' % ev)
    #         print('NCCF_CAL_REFBIAS_V: ', model.get_sadc_value(0, nccf_cal_ref_g1_pg0_address))
    # except KeyboardInterrupt:
    #     pass
    #
    # board.enable_output(BusId.sADC, False)
    #
    # # Note that if you want all sADC values, then you can do something like the following:
    # sadc_values = model.get_sadc_values(0)
    # for i, v in enumerate(sadc_values):
    #     print('%30s: %d' % (get_sadc_description(i), v))

    time.sleep(0.1)

    config = model.get_configuration()

    set_parameter(config.chips[0].shared_parameters01, 'LBWR_VB_P', 3, 255)
    set_parameter(config.chips[0].shared_parameters23, 'LBWR_VB_P', 3, 255)
    set_parameter(config.chips[0].shared_parameters01, 'NCCF_CAL_OFFBIAS_P', 0, 0)
    set_parameter(config.chips[0].shared_parameters23, 'NCCF_CAL_OFFBIAS_P', 0, 0)
    # Disable all other sADC channels - this is optional
    for core in config.chips[0].cores:
        core.sadc_enables.soif_mem = True
        core.sadc_enables.soif_refractory = True
        core.sadc_enables.soad_dpi = True
        core.sadc_enables.soca_dpi = True
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
    config.chips[0].sadc_enables.nccf_extin_vi_group0_pg1 = True
    config.chips[0].sadc_enables.nccf_cal_refbias_v_group1_pg1 = True
    config.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg1 = True
    config.chips[0].sadc_enables.nccf_extin_vi_group2_pg1 = True
    config.chips[0].sadc_enables.nccf_extin_vi_group0_pg0 = True
    config.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg0 = True
    config.chips[0].sadc_enables.nccf_extin_vi_group2_pg0 = True
    model.apply_configuration(config)
    time.sleep(0.1)

    model.set_sadc_sample_period_ms(100)
    board.enable_output(BusId.sADC, True)

    fine_step = 8
    rate = np.zeros((64, 7, int(256/fine_step)))

    regimeHigh = 0

    for coarse in range(7):
        print(f"coarse = {coarse}")
        if coarse == 0:
            for i in range(3):
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_PWLK_P", 3, 160)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_PWLK_P", 3, 160)
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_HYS_P", 0, 0)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_HYS_P", 0, 0)
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_BIAS_P", 5, 200)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_BIAS_P", 5, 200)
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_REF_L_V", 0, 100)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_REF_L_V", 0, 100)
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_REF_H_V", 3, 80)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_REF_H_V", 3, 80)
        model.apply_configuration(config)
        time.sleep(0.1)

        if coarse == 3:
            regimeHigh = 1
            for i in range(3):
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_PWLK_P", 5, 255)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_PWLK_P", 5, 255)
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_HYS_P", 0, 0)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_HYS_P", 0, 0)
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_BIAS_P", 3, 40)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_BIAS_P", 3, 40)
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_REF_L_V", 0, 100)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_REF_L_V", 0, 100)
                set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_REF_H_V", 5, 250)
                set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_REF_H_V", 5, 250)
        model.apply_configuration(config)
        time.sleep(0.1)

        for f in range(int(256 / fine_step)):
            fine = f * fine_step
            print(f"fine = {fine}")
            set_parameter(config.chips[0].shared_parameters01, "NCCF_CAL_OFFBIAS_P", coarse - regimeHigh, fine)
            set_parameter(config.chips[0].shared_parameters23, "NCCF_CAL_OFFBIAS_P", coarse - regimeHigh, fine)
            model.apply_configuration(config)
            time.sleep(0.25)

            sadc_values = model.get_sadc_values(0)
            for i, v in enumerate(sadc_values):
                print('%30s: %d' % (get_sadc_description(i), v))
                rate[i][coarse][f] = v

    # for i in range(rate.shape[0]):
    #     for c in range(rate.shape[1]):
    #         print(rate[i][c])
    np.save('sadc_rates', rate)


def test_adaptation(board, number_of_chips):
    
    model = board.get_model()
    model.reset(ResetType.ConfigReset, 0b1)

    time.sleep(0.1)
    config = model.get_configuration()
    
    neuron = 0

    set_parameter(config.chips[0].global_parameters, "R2R_BUFFER_AMPB", 5, 255)
    set_parameter(config.chips[0].global_parameters, "R2R_BUFFER_CCB", 5, 255)
    set_parameter(config.chips[0].shared_parameters01, "LBWR_VB_P", 5, 255)
    set_parameter(config.chips[0].shared_parameters23, "LBWR_VB_P", 5, 255)
    for c in range(4):
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_GAIN_N", 4, 50)
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 50)
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_REFR_N", 5, 255)
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_DC_P", 3, 120)
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 5, 40)
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_CC_N", 5, 255)

    for h in range(number_of_chips):
        for c in range(4):
            config.chips[h].cores[c].neuron_monitoring_on = True
            config.chips[h].cores[c].monitored_neuron = neuron  # monitor neuron 3 on each core
    model.apply_configuration(config)
    time.sleep(0.1)

    for c in range(4):
        set_parameter(config.chips[0].cores[c].parameters, "SOAD_PWTAU_N", 4, 255)
        set_parameter(config.chips[0].cores[c].parameters, "SOAD_GAIN_P", 3, 50)
        set_parameter(config.chips[0].cores[c].parameters, "SOAD_TAU_P", 0, 10)
        set_parameter(config.chips[0].cores[c].parameters, "SOAD_W_N", 3, 50)
        set_parameter(config.chips[0].cores[c].parameters, "SOAD_CASC_P", 4, 80)
        set_parameter(config.chips[0].cores[c].parameters, "SOCA_W_N", 3, 160)
        set_parameter(config.chips[0].cores[c].parameters, "SOCA_GAIN_P", 3, 100)
        set_parameter(config.chips[0].cores[c].parameters, "SOCA_TAU_P", 0, 30)
    model.apply_configuration(config)
    time.sleep(0.1)

    for c in range(4):
        # config.chips[0].cores[c].neurons[neuron].latch_so_dc = True
        config.chips[0].cores[c].neurons[neuron].latch_so_adaptation = True
        model.apply_configuration(config)
        time.sleep(0.1)

    set_parameter(config.chips[0].shared_parameters01, "NCCF_CAL_OFFBIAS_P", 1, 255)
    set_parameter(config.chips[0].shared_parameters23, "NCCF_CAL_OFFBIAS_P", 1, 255)
    for core in config.chips[0].cores:
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
    config.chips[0].sadc_enables.nccf_extin_vi_group0_pg1 = True
    config.chips[0].sadc_enables.nccf_cal_refbias_v_group1_pg1 = True
    config.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg1 = True
    config.chips[0].sadc_enables.nccf_extin_vi_group2_pg1 = True
    config.chips[0].sadc_enables.nccf_extin_vi_group0_pg0 = True
    config.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg0 = True
    config.chips[0].sadc_enables.nccf_extin_vi_group2_pg0 = True
    model.apply_configuration(config)
    time.sleep(0.1)

    for i in range(3):
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_PWLK_P", 5, 160)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_PWLK_P", 5, 160)
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_HYS_P", 0, 0)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_HYS_P", 0, 0)
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_BIAS_P", 5, 200)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_BIAS_P", 5, 200)
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_REF_L_V", 0, 100)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_REF_L_V", 0, 100)
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_REF_H_V", 3, 80)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_REF_H_V", 3, 80)
    model.apply_configuration(config)
    time.sleep(0.1)

    clear_srams(config=config, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    model.apply_configuration(config)
    time.sleep(0.1)

    model.set_sadc_sample_period_ms(5)
    board.enable_output(BusId.sADC, True)
    board.enable_output(BusId.W, True)
    for i in range(20):
        config.chips[0].cores[0].neurons[neuron].latch_so_dc = True
        model.apply_configuration(config)
        for _ in range(50):
            sadcValues = model.get_sadc_values(0)
            # print(f"{sadcValues[1]}, {sadcValues[6]},{sadcValues[8]}")
            # print(f"{get_sadc_description(33)}: {sadcValues[33]}; {get_sadc_description(38)}: {sadcValues[38]}")
            print(f"{sadcValues[33]}, {sadcValues[38]}")
            time.sleep(0.005)
        config.chips[0].cores[0].neurons[neuron].latch_so_dc = False
        model.apply_configuration(config)
        for _ in range(100):
            sadcValues = model.get_sadc_values(0)
            # print(f"{sadcValues[1]}, {sadcValues[6]},{sadcValues[8]}")
            # print(f"{get_sadc_description(33)}: {sadcValues[33]}; {get_sadc_description(38)}: {sadcValues[38]}")
            print(f"{sadcValues[33]}, {sadcValues[38]}")
            time.sleep(0.005)
    send_events(board=board, events=[], min_delay=1000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)

def test_stp(board, number_of_chips):
    model = board.get_model()
    model.reset(ResetType.ConfigReset, 0b1)

    time.sleep(1)
    config = model.get_configuration()

    neuron = 0
    syn_mon = 20
    tag = 1024

    set_parameter(config.chips[0].global_parameters, "R2R_BUFFER_AMPB", 5, 255)
    set_parameter(config.chips[0].global_parameters, "R2R_BUFFER_CCB", 5, 255)
    set_parameter(config.chips[0].shared_parameters01, "SYAM_STDWAMPB", 5, 255)
    set_parameter(config.chips[0].shared_parameters01, "SYAM_STDWCCB", 5, 255)
    set_parameter(config.chips[0].shared_parameters01, "LBWR_VB_P", 5, 255)
    set_parameter(config.chips[0].shared_parameters23, "LBWR_VB_P", 5, 255)
    for c in range(1):
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_GAIN_N", 2, 100)
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_LEAK_N", 2, 50)
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_REFR_N", 3, 254)
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(config.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(config.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 80)
        set_parameter(config.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 4, 160)
        set_parameter(config.chips[0].cores[c].parameters, 'SYAM_W0_P', 4, 40)
        set_parameter(config.chips[0].cores[c].parameters, 'SYAM_W1_P', 4, 60)
        set_parameter(config.chips[0].cores[c].parameters, 'SYAM_W2_P', 4, 80)
        set_parameter(config.chips[0].cores[c].parameters, 'SYAM_W3_P', 4, 120)
        set_parameter(config.chips[0].cores[c].parameters, 'SYAM_STDW_N', 4, 80)
        set_parameter(config.chips[0].cores[c].parameters, 'SYAW_STDSTR_N', 0, 80)
        set_parameter(config.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 50)
    model.apply_configuration(config)
    time.sleep(0.1)

    for c in range(1):
        config.chips[0].cores[c].neuron_monitoring_on = True
        config.chips[0].cores[c].monitored_neuron = neuron
        config.chips[0].cores[c].enable_pulse_extender_monitor1 = True
        config.chips[0].cores[c].enable_syaw_stdbuf_an = True
    model.apply_configuration(config)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        cam_exc = [Dynapse2Synapse() for _ in range(64)]
        weights = [True, False, False, False]
        cam_exc[syn_mon].weight = weights
        cam_exc[syn_mon].tag = tag
        cam_exc[syn_mon].dendrite = Dendrite.ampa
        cam_exc[syn_mon].stp = True
        config.chips[0].cores[c].neurons[neuron].synapses = cam_exc
    model.apply_configuration(config)
    time.sleep(0.1)

    clear_srams(config=config, neurons=range(256), cores=range(1), chips=range(number_of_chips))
    model.apply_configuration(config)
    time.sleep(0.1)

    set_parameter(config.chips[0].shared_parameters01, "NCCF_CAL_OFFBIAS_P", 1, 255)
    set_parameter(config.chips[0].shared_parameters23, "NCCF_CAL_OFFBIAS_P", 1, 255)
    for core in config.chips[0].cores:
        core.sadc_enables.soif_mem = True
        #core.sadc_enables.soif_refractory = True
        core.sadc_enables.soad_dpi = True
        core.sadc_enables.soca_dpi = True
        core.sadc_enables.deam_edpi = False
        core.sadc_enables.deam_idpi = True
        core.sadc_enables.denm_edpi = True
        core.sadc_enables.denm_idpi = True
        core.sadc_enables.dega_idpi = True
        core.sadc_enables.desc_idpi = True
        core.sadc_enables.sy_w42 = True
        core.sadc_enables.sy_w21 = False
        core.sadc_enables.soho_sogain = True
        core.sadc_enables.soho_degain = True
    config.chips[0].sadc_enables.nccf_extin_vi_group0_pg1 = True
    config.chips[0].sadc_enables.nccf_cal_refbias_v_group1_pg1 = True
    config.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg1 = True
    config.chips[0].sadc_enables.nccf_extin_vi_group2_pg1 = True
    config.chips[0].sadc_enables.nccf_extin_vi_group0_pg0 = True
    config.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg0 = True
    config.chips[0].sadc_enables.nccf_extin_vi_group2_pg0 = True
    model.apply_configuration(config)
    time.sleep(0.1)

    for i in range(3):
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_PWLK_P", 5, 255)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_PWLK_P", 5, 255)
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_HYS_P", 0, 0)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_HYS_P", 0, 0)
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_BIAS_P", 3, 40)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_BIAS_P", 3, 40)
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_REF_L_V", 0, 100)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_REF_L_V", 0, 100)
        set_parameter(config.chips[0].sadc_group_parameters01[i], "NCCF_REF_H_V", 5, 250)
        set_parameter(config.chips[0].sadc_group_parameters23[i], "NCCF_REF_H_V", 5, 250)
    model.apply_configuration(config)
    time.sleep(0.1)



    ts = get_fpga_time(board=board) + 100000
    dt = 6000
    input_events = []
    for i in range(100):
        for j in range(10):
            input_events += [
                AerConstructor(DestinationConstructor(tag=tag,
                                                      core=[True, False, False, False]).destination,
                               ts).aer,
                AerConstructor(DestinationConstructor(tag=tag,
                                                      core=[True, False, False, False], x_hop=-7).destination,
                               ts).aer]
            ts += dt
        ts += 40 * dt

    print("\nAll configurations done!\n")

    model.set_sadc_sample_period_ms(5)
    board.enable_output(BusId.sADC, True)
    board.enable_output(BusId.W, True)

    send_events(board=board, events=input_events, min_delay=100000)

    # print("\nAll spikes sent!\n")


    while True:
        sadcValues = model.get_sadc_values(0)
        time.sleep(0.005)
        # for _ in range(300):
        #     time.sleep(0.00001)
            # for ev in board.read_events():
            #     if ev.event.tag < 1024:
            #         print(ev.event.tag)
                # pass
        # print(f"{get_sadc_description(42)}: {sadcValues[42]}; {get_sadc_description(57)}: {sadcValues[57]}")
        print(f"{sadcValues[42]}, {sadcValues[57]}")

