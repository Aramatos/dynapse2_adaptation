import time
import sys
import os
import json

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_spikegen import *
from lib.dynapse2_raster import *
from samna.dynapse2 import *
import random

default_tags = [[[random.randint(0, 2047) for synapse in range(64)] for neuron in range(256)] for core in range(4)]


def test_dc(board, number_of_chips):
    # random.seed(2)
    neurons = range(10)
    synapses = range(10, 11)

    model = board.get_model()
    myConfig = model.get_configuration()

    for source_cores in [[0]]:
        for target_cores in [[2]]:
            # for neurons in [[i] for i in range(15, 241, 15)]:

            print(source_cores)
            print(target_cores)
            print(neurons)

            # model.reset(ResetType.ConfigReset, (1 << number_of_chips) - 1)
            # time.sleep(1)

            model.reset(ResetType.PowerCycle, (1 << number_of_chips) - 1)
            time.sleep(1)

            model.apply_configuration(myConfig)
            time.sleep(1)

            # set neuron parameters
            print("Setting parameters")
            set_parameter(myConfig.chips[0].global_parameters, "R2R_BUFFER_AMPB", 5, 255)
            set_parameter(myConfig.chips[0].global_parameters, "R2R_BUFFER_CCB", 5, 255)
            set_parameter(myConfig.chips[0].shared_parameters01, "LBWR_VB_P", 5, 255)
            set_parameter(myConfig.chips[0].shared_parameters23, "LBWR_VB_P", 5, 255)
            set_parameter(myConfig.chips[0].shared_parameters01, "PG_BUF_N", 1, 50)
            set_parameter(myConfig.chips[0].shared_parameters23, "PG_BUF_N", 1, 50)
            for h in range(number_of_chips):
                for c in range(4):
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY0_P", 1, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY1_P", 1, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY2_P", 1, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_ITAU_P", 4, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DENM_ETAU_P", 4, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DENM_ITAU_P", 4, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DEGA_ITAU_P", 4, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DESC_ITAU_P", 4, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SOHO_VB_P", 1, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SOAD_PWTAU_N", 1, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SOAD_TAU_P", 1, 255)
                    #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SOCA_TAU_P", 1, 255)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYSA_VRES_N", 5, 254)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYSA_VB_P", 5, 254)
                for c in source_cores + target_cores:
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_LEAK_N", 0, 255)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_SPKTHR_P", 3, 255)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_GAIN_N", 3, 255)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_REFR_N", 1, 255)
                for c in source_cores:
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_DC_P", 1, 160)
                for c in target_cores:
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W0_P", 5, 200)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W1_P", 0, 0)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_ETAU_P", 3, 40)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_EGAIN_P", 4, 80)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_EXT_N", 3, 255)
                    # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY0_P", 3, 255)
                    # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY1_P", 3, 255)
                    # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY2_P", 3, 255)

            model.apply_configuration(myConfig)
            time.sleep(1)

            # set neurons to monitor
            print("Setting monitors")
            for h in range(number_of_chips):
                for c in source_cores + target_cores:
                    myConfig.chips[h].cores[c].neuron_monitoring_on = True
                    myConfig.chips[h].cores[c].monitored_neuron = neurons[-1]  # monitor neuron 3 on each core
                    myConfig.chips[0].cores[target_cores[-1]].enable_pulse_extender_monitor1 = True
            model.apply_configuration(myConfig)
            time.sleep(1)

            # for i in range(3):
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_PWLK_P", 5, 255)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_PWLK_P", 5, 255)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_HYS_P", 0, 0)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_HYS_P", 0, 0)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_BIAS_P", 0, 40)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_BIAS_P", 0, 40)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_REF_L_V", 0, 100)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_REF_L_V", 0, 100)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_REF_H_V", 5, 250)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_REF_H_V", 5, 250)
            #
            #
            # model.apply_configuration(myConfig)
            # time.sleep(0.1)
            #
            # set_parameter(myConfig.chips[0].shared_parameters01, "NCCF_CAL_OFFBIAS_P", 0, 0)
            # set_parameter(myConfig.chips[0].shared_parameters23, "NCCF_CAL_OFFBIAS_P", 0, 0)
            # set_parameter(myConfig.chips[0].shared_parameters01, "NCCF_CAL_REFBIAS_V", 5, 255)
            # set_parameter(myConfig.chips[0].shared_parameters23, "NCCF_CAL_REFBIAS_V", 5, 255)
            # for core in myConfig.chips[0].cores:
            #     core.sadc_enables.soif_mem = True
            #     core.sadc_enables.soif_refractory = True
            #     core.sadc_enables.soad_dpi = True
            #     core.sadc_enables.soca_dpi = True
            #     core.sadc_enables.deam_edpi = True
            #     core.sadc_enables.deam_idpi = True
            #     core.sadc_enables.denm_edpi = True
            #     core.sadc_enables.denm_idpi = True
            #     core.sadc_enables.dega_idpi = True
            #     core.sadc_enables.desc_idpi = True
            #     core.sadc_enables.sy_w42 = True
            #     core.sadc_enables.sy_w21 = True
            #     core.sadc_enables.soho_sogain = True
            #     core.sadc_enables.soho_degain = True
            # myConfig.chips[0].sadc_enables.nccf_extin_vi_group0_pg1 = True
            # myConfig.chips[0].sadc_enables.nccf_cal_refbias_v_group1_pg1 = True
            # myConfig.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg1 = True
            # myConfig.chips[0].sadc_enables.nccf_extin_vi_group2_pg1 = True
            # myConfig.chips[0].sadc_enables.nccf_extin_vi_group0_pg0 = True
            # myConfig.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg0 = True
            # myConfig.chips[0].sadc_enables.nccf_extin_vi_group2_pg0 = True
            # model.apply_configuration(myConfig)
            # time.sleep(1)

            # set neuron latches to get DC input
            print("Setting DC")
            set_dc_latches(config=myConfig, neurons=neurons, cores=source_cores, chips=range(number_of_chips))
            # set_type_latches(config=myConfig, neurons=neurons, cores=range(4), chips=range(number_of_chips))
            # set_type_latches(config=myConfig, neurons=neurons, cores=range(1), chips=range(number_of_chips))
            model.apply_configuration(myConfig)
            time.sleep(1)

            # set CAM -- synapses
            for c in target_cores:
                for i in range(256):
                    cam_exc = [Dynapse2Synapse() for _ in range(64)]
                    for j in range(64):
                        if i in neurons and j in synapses:
                            # if j != 7:
                            weights = [True, False, False, False]
                            cam_exc[j].weight = weights
                            # cam_exc[j].tag = random.choice(source_cores) * 256 + random.choice(neurons)
                            cam_exc[j].tag = 1024 + source_cores[0] * 256 + i
                            cam_exc[j].dendrite = Dendrite.ampa
                        else:
                            # if j != 7:
                            weights = [False, True, False, False]
                            cam_exc[j].weight = weights
                            cam_exc[j].tag = 0
                            cam_exc[j].dendrite = Dendrite.none
                    myConfig.chips[0].cores[c].neurons[i].synapses = cam_exc
            model.apply_configuration(myConfig)
            time.sleep(1)

            # set SRAM -- axons
            print("Setting SRAMs")
            clear_srams(config=myConfig, neurons=neurons, source_cores=source_cores, cores=target_cores,
                        chips=range(number_of_chips), all_to_all=True)
            model.apply_configuration(myConfig)
            time.sleep(1)

            print("\nAll configurations done!\n")
            send_events(board=board, events=[], min_delay=10000000)
            output_events = [[], []]
            get_events(board=board, extra_time=100, output_events=output_events)

            for c in source_cores:
                set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 0)
                for n in neurons:
                    myConfig.chips[0].cores[c].neurons[n].latch_so_dc = False
            for c in source_cores + target_cores:
                for n in neurons:
                    myConfig.chips[0].cores[c].neurons[n].destinations = [Dynapse2Destination()] * 4
            model.apply_configuration(myConfig)
            time.sleep(1)

            while len(board.read_events()) > 0:
                pass

            spike_count(output_events=output_events)
            plot_raster(output_events=output_events)
            # while True:
            #     pass

    # model.set_sadc_sample_period_ms(100)
    # board.enable_output(BusId.sADC, True)
    # board.enable_output(BusId.W, True)

    # print("\nAll spikes sent!\n")

    # for _ in range(100):
    #     # sadcValues = model.get_sadc_values(0)
    #     time.sleep(0.1)
    #     # for _ in range(300):
    #     #     time.sleep(0.00001)
    #     # for ev in board.read_events():
    #     #     if ev.event.tag < 1024:
    #     #         print(ev.event.tag)
    #     # pass
    #     # print(f"{get_sadc_description(42)}: {sadcValues[42]}; {get_sadc_description(57)}: {sadcValues[57]}")
    #     # print(f"{sadcValues[42]}, {sadcValues[57]}")
    #     sadc_values = model.get_sadc_values(0)
    #     for i, v in enumerate(sadc_values):
    #         print('%30s: %d' % (get_sadc_description(i), v))

    # set_type_latches(config=myConfig, neurons=neurons, cores=range(1), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(1)

    # for key in myConfig.chips[0].cores[0].parameters.keys():
    # for key in myConfig.chips[0].shared_parameters01.keys():
    #     print(key)
    #     set_parameter(myConfig.chips[0].shared_parameters01, key, 5, 255)
    #     set_parameter(myConfig.chips[0].shared_parameters23, key, 5, 255)
    #     model.apply_configuration(myConfig)
    #     time.sleep(5)

    # for h in range(number_of_chips):
    #     for c in range(4):
    #         set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_DC_P", 0, 0)
    #
    # model.apply_configuration(myConfig)
    # time.sleep(1)


def initialize(board, number_of_chips):
    model = board.get_model()
    myConfig = model.get_configuration()

    model.reset(ResetType.PowerCycle, (1 << number_of_chips) - 1)
    time.sleep(1)

    model.apply_configuration(myConfig)
    time.sleep(5)

    # set neuron parameters
    print("Setting parameters")
    set_parameter(myConfig.chips[0].global_parameters, "R2R_BUFFER_AMPB", 5, 255)
    set_parameter(myConfig.chips[0].global_parameters, "R2R_BUFFER_CCB", 5, 255)
    set_parameter(myConfig.chips[0].shared_parameters01, "LBWR_VB_P", 5, 255)
    set_parameter(myConfig.chips[0].shared_parameters23, "LBWR_VB_P", 5, 255)
    set_parameter(myConfig.chips[0].shared_parameters01, "PG_BUF_N", 0, 100)
    set_parameter(myConfig.chips[0].shared_parameters23, "PG_BUF_N", 0, 100)
    for h in range(number_of_chips):
        for c in range(4):
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYSA_VRES_N", 5, 254)
            if c == 1:
                set_parameter(myConfig.chips[h].cores[c].parameters, "SYSA_VB_P", 5, 254)
            else:
                set_parameter(myConfig.chips[h].cores[c].parameters, "SYSA_VB_P", 5, 254)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_LEAK_N", 0, 255)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_SPKTHR_P", 4, 80)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_GAIN_N", 3, 255)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_REFR_N", 4, 255)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W0_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W1_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W2_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W3_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_ETAU_P", 5, 80)
            set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_EGAIN_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_EXT_N", 4, 40)
    model.apply_configuration(myConfig)
    time.sleep(1)

    return model, myConfig


def sweep_cams(board, model, myConfig, cores, neurons, synapse, weight, tag,
               cam_list, px_list, good_list, double=True, plot=False):
    # print(f"cores {cores}, neurons {neurons}, synapse {synapse}, weight {weight}, tag {tag}")
    # for c in cores:
    #     for i in range(256):
    #         myConfig.chips[0].cores[c].neurons[i].latch_soif_kill = False
    # print(tag)

    # set CAM -- synapses
    for c in range(4):
        for i in range(256):
            cam_exc = [Dynapse2Synapse() for _ in range(64)]
            for s in range(64):
                if c in cores and i in neurons[c] and s == synapse[c][i]:
                    cam_exc[s].weight = [_ == weight[c][i] for _ in range(4)]
                    cam_exc[s].tag = tag[c][i]
                    cam_exc[s].dendrite = Dendrite.ampa
                else:
                    cam_exc[s].tag = default_tags[c][i][s]
            myConfig.chips[0].cores[c].neurons[i].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(.1)

    input_events = []
    ts = get_fpga_time(board=board) + 100000
    for c in cores:
        for i in neurons[c]:
            input_events += [AerConstructor(DestinationConstructor(
                tag=tag[c][i], core=[_ == c for _ in range(4)]).destination, ts).aer]
            ts += 100
            if double:
                input_events += [AerConstructor(
                    DestinationConstructor(tag=tag[c][i], core=[_ == c for _ in range(4)]).destination, ts).aer]
                ts += 100
        ending_tag = 0
        for i in range(1, 9):
            while ending_tag in tag[c]:
                ending_tag += 1
            input_events += [AerConstructor(
                DestinationConstructor(tag=ending_tag, core=[_ == c for _ in range(4)]).destination, ts).aer]
            ts += 100

    send_events(board=board, events=input_events, min_delay=0)

    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    counts = spike_count(output_events=output_events, show=True)

    while True:
        residual_events = [[], []]
        send_events(board=board, events=[], min_delay=0)
        get_events(board=board, extra_time=100, output_events=residual_events)
        residual_counts = spike_count(output_events=residual_events, show=False)
        if sum(residual_counts) > 1:
            print(sum(residual_counts))
            for c in range(4):
                if sum(residual_counts[c * 256: c * 256 + 256]):
                    send_events(board=board, events=[AerConstructor(
                        DestinationConstructor(tag=2047, core=[_ == c for _ in range(4)]).destination, ts).aer],
                                min_delay=1000)
        else:
            break

    if len(counts):
        for c in cores:
            for i in neurons[c]:
                s = synapse[c][i]
                t = myConfig.chips[0].cores[c].neurons[i].synapses[s].tag
                if counts[c * 256 + i] < 1 + double:
                    cam_list += [[c, i, s, t]]
                    if double and counts[c * 256 + i] == 0:
                        cam_list += [[c, i, s, t]]
                elif counts[c * 256 + i] > 4:
                    px_list += [[c, i, s, t]]
                    print(c, i, s, t)
                    if counts[c * 256 + i] > 100:
                        input_events += [AerConstructor(
                            DestinationConstructor(tag=2047, core=[_ == c for _ in range(4)]).destination,
                            ts).aer]
                else:
                    good_list += [[c, i, s, t]]

    if plot:
        plot_raster(output_events=output_events)


def compute_tag_candidates(old_tag_candidates, px_list):
    bad_px = [[_ in px_list[px_list[:, 0] == core, 3] or
               _ not in old_tag_candidates[core][0] + old_tag_candidates[core][1]
               for _ in range(2048)] for core in range(4)]
    while True:
        bad_px_filled = [[False] * 2048 for _ in range(4)]
        for core, bad_px_core in enumerate(bad_px):
            for i in range(2048):
                for j in range(3, 11):
                    flag_j = bad_px_core[i] * 15 + bad_px_core[i ^ (1 << j)]
                    if core == 2 and i == 0:
                        print(bad_px_core[i], bad_px_core[i ^ (1 << j)], flag_j)
                    for k in range(2, j):
                        flag_k = flag_j + bad_px_core[i ^ (1 << k)] + bad_px_core[i ^ (1 << j) ^ (1 << k)]
                        for l in range(1, k):
                            flag_l = flag_k + bad_px_core[i ^ (1 << l)] + bad_px_core[i ^ (1 << j) ^ (1 << l)] + \
                                     bad_px_core[i ^ (1 << k) ^ (1 << l)] + bad_px_core[
                                         i ^ (1 << j) ^ (1 << k) ^ (1 << l)]
                            for m in range(l):
                                flag = flag_l + bad_px_core[i ^ (1 << m)] + \
                                       bad_px_core[i ^ (1 << j) ^ (1 << m)] + \
                                       bad_px_core[i ^ (1 << k) ^ (1 << m)] + \
                                       bad_px_core[i ^ (1 << j) ^ (1 << k) ^ (1 << m)] + \
                                       bad_px_core[i ^ (1 << l) ^ (1 << m)] + \
                                       bad_px_core[i ^ (1 << j) ^ (1 << l) ^ (1 << m)] + \
                                       bad_px_core[i ^ (1 << k) ^ (1 << l) ^ (1 << m)] + \
                                       bad_px_core[i ^ (1 << j) ^ (1 << k) ^ (1 << l) ^ (1 << m)]
                                if flag > 14:
                                    break
                            if flag > 14:
                                break
                        if flag > 14:
                            break
                    if flag > 14:
                        bad_px_filled[core][i] = True
                        break
        if np.sum(bad_px != bad_px_filled):
            bad_px = [[_ for _ in bad_px_core] for core, bad_px_core in enumerate(bad_px_filled)]
        else:
            break
    return [[[_ for _ in range(1024) if not bad_px_core[_]],
             [_ for _ in range(1024, 2048) if not bad_px_core[_]]] for core, bad_px_core in enumerate(bad_px)]


def compute_profile(px_filename, cam_filename):
    data_px = np.loadtxt(px_filename, delimiter=",", dtype=int)
    data_cams = np.loadtxt(cam_filename, delimiter=",", dtype=int)

    bad_neuron = np.zeros((4, 256), dtype=bool)
    bad_tag = np.zeros((4, 1024), dtype=bool)
    cam_bad_tag = np.zeros((4, 1024), dtype=bool)
    bad_neuron_tag = np.zeros((4, 256, 1024), dtype=bool)

    for core_neuron_synapse_tag in data_px:
        core = core_neuron_synapse_tag[0]
        neuron = core_neuron_synapse_tag[1]
        tag = core_neuron_synapse_tag[3] - 1024
        bad_neuron[core, neuron] = True
        bad_tag[core, tag] = True
        bad_neuron_tag[core, neuron, tag] = True
    good_neuron = 1 - bad_neuron
    good_tag = 1 - bad_tag

    # for core in range(4):
    #     for tag, failed in enumerate(np.bincount((data_cams[data_cams[:, 0] == core, 3]).astype("int"))):
    #         if failed > 16: # if failed for at least 16 out of 64
    #             cam_bad_tag[core, tag] = True
    # good_cam_tag = 1 - cam_bad_tag

    bad_cams = np.zeros((4, 256, 64), dtype=bool)
    for core_neuron_synapse_tag in data_cams:
        core = core_neuron_synapse_tag[0]
        neuron = core_neuron_synapse_tag[1]
        synapse = core_neuron_synapse_tag[2]
        tag = core_neuron_synapse_tag[3] - 1024
        if neuron in good_neuron[core]:
            if good_tag[core][tag]:  # and good_cam_tag[core][tag]:
                bad_cams[core, neuron, synapse] = True
        else:
            # if good_cam_tag[core][tag]:
            bad_cams[core, neuron, synapse] = True

    # valid_synapses = [[list(np.nonzero(1 - bad_cams[core, neuron, :])[0]) for neuron in range(256)] for core in range(4)]

    with open('good_neurons.json', 'w') as f:
        json.dump([[neuron for neuron in range(256) if good_neuron[core][neuron]] for core in range(4)], f)
    with open('good_tags.json', 'w') as f:
        json.dump([[tag for tag in range(1024) if not bad_tag[core][tag]] for core in range(4)], f)

    return good_neuron, good_tag  # , good_cam_tag, valid_synapses


def write_cam_profile(good_neuron, good_cam_tag, data_cams, total_num_all, total_num_good):
    cams_matrix = np.zeros((4, 256, 64))
    for core_neuron_synapse_tag in data_cams:
        if good_cam_tag[core_neuron_synapse_tag[0]][core_neuron_synapse_tag[3]]:
            cams_matrix[core_neuron_synapse_tag[0], core_neuron_synapse_tag[1], core_neuron_synapse_tag[2]] += 1
    for core in range(4):
        for neuron in range(256):
            if good_neuron[core][neuron]:
                cams_matrix[core][neuron] /= total_num_good[core]
            else:
                cams_matrix[core][neuron] /= total_num_all[core]

    with open('cams_matrix.json', 'w') as f:
        json.dump([[[cams for cams in cams_neuron] for cams_neuron in cams_core] for cams_core in cams_matrix], f)


def test_cams_two_steps(board, number_of_chips, board_name="pink"):
    model, myConfig = initialize(board=board, number_of_chips=number_of_chips)

    px_filename = "bad_pxs_" + board_name + ".csv"
    cam_filename = "bad_cams_" + board_name + ".csv"
    good_filename = "good_" + board_name + ".csv"

    # tag_candidates = [[list(range(1024)), list(range(1024, 2048))],
    #                   [list(range(512, 1024)), list(range(1024, 2048))],
    #                   [[], [1032 + i * 16 + j for i in range(64) for j in range(8)]],
    #                   [list(range(1024)), list(range(1280, 2048))]]
    # tag_candidates = [[[], list(range(1024, 2048))],
    #                   [list(range(512, 768)) + list(range(896, 1024)), list(range(1024, 2048))],
    #                   [[], [1032 + k * 128 + i * 16 + j for k in range(8) for i in range(6) for j in range(8) if (8 * k + i) % 32 != 12]],
    #                   [list(range(1024)), list(range(1280, 2048))]]
    # tag_candidates = [[list(range(1024)), list(range(1024, 2048))],
    #                   [[], []],
    #                   [[], []],
    #                   [list(range(1024)), list(range(1280, 2048))]]

    tag_candidates = [[list(range(1024)), list(range(1024, 2048))],
                      [list(range(1024)), list(range(1024, 2048))],
                      [list(range(1024)), list(range(1024, 2048))],
                      [list(range(1024)), list(range(1024, 2048))]]
    # tag_candidates = [[[], list(range(1024, 2048))],
    #                   [list(range(512, 1024)), list(range(1024, 2048))],
    #                   [list(range(1024)), list(range(1024, 2048))],
    #                   [list(range(256, 1024)), list(range(1600, 1664)) + list(range(1728, 2048))]]

    identify_tag_candidates = True

    while True:

        tag_counts = [[0] * 4 for _ in range(2)]

        # if not os.path.exists(px_filename) or not os.path.exists(cam_filename):
        if True:

            print("Test all cams first")

            px_list = []
            cam_list = []
            good_list = []

            for i in range(4):
                print(f"round {i}")

                # if not identify_tag_candidates:
                clear_srams(config=myConfig, neurons=range(256), source_cores=[], cores=range(4),
                            chips=range(number_of_chips), all_to_all=True, monitor_cam=i)
                model.apply_configuration(myConfig)
                time.sleep(1)

                synapse_shuffle = np.array([[random.sample(range(64), k=64) for _ in range(256)] for __ in range(4)])
                synapse_round = 0

                for j in range(16):
                    print(f"trial {j}")

                    # if identify_tag_candidates:
                    #     clear_srams(config=myConfig, neurons=range(256), source_cores=[], cores=range(4),
                    #                 chips=range(number_of_chips), all_to_all=True, monitor_cam=j % 4)
                    #     model.apply_configuration(myConfig)
                    #     time.sleep(0.1)

                    weight = np.array([[random.randint(0, 3) for _ in range(256)] for __ in range(4)])

                    for tag_round in range(8):
                        print(tag_counts)
                    # for tag_offset in range(0, 2048, 256):
                        # for tag_offset in range(0, 1024, 256):
                        print(f"synapse round {synapse_round}")

                        synapse = synapse_shuffle[:, :, synapse_round]
                        # tag = [[(__ * 256 + _ + tag_offset) % 2048 for _ in random.sample(range(256), k=256)] for __ in
                        #        range(4)]
                        # tag = [[((__ * 256 + _ + tag_offset) % 1024) + 1024 for _ in random.sample(range(256), k=256)] for __ in range(4)]
                        msb = tag_round % 2
                        tag = [random.sample([tag_candidates[core][msb][tag_id % len(tag_candidates[core][msb])]
                                for tag_id in range(tag_counts[msb][core], tag_counts[msb][core] + 256)], k=256)
                               if len(tag_candidates[core][msb]) else [] for core in range(4)]
                        tag_counts[msb] = [(count + 256) % len(tag_candidates[core][msb])
                                      if len(tag_candidates[core][msb]) else 0 for core, count in enumerate(tag_counts[msb])]
                        synapse_round = synapse_round + (msb)
                        sweep_cams(board=board, model=model, myConfig=myConfig,
                                   cores=[core for core in range(4) if len(tag_candidates[core][msb])],
                                   neurons=[random.sample(range(256), k=256) for _ in range(4)],
                                   synapse=synapse, weight=weight, tag=tag,
                                   cam_list=cam_list, px_list=px_list, good_list=good_list,
                                   double=True, plot=False)
                        if len(px_list):
                            break
                    if len(px_list):
                        break
                if len(px_list):
                    break

        if len(px_list) == 0:
            break

        tag_candidates = compute_tag_candidates(old_tag_candidates=tag_candidates, px_list=np.asarray(px_list).astype(int))
        print(tag_candidates)

    np.savetxt(px_filename, np.asarray(px_list).astype(int), fmt='%i', delimiter=",")
    np.savetxt(cam_filename, np.asarray(cam_list).astype(int), fmt='%i', delimiter=",")
    np.savetxt(good_filename, np.asarray(good_list).astype(int), fmt='%i', delimiter=",")

    # compute_profile(px_filename, cam_filename)
    # good_neuron, good_tag, good_cam_tag, valid_synapses = compute_profile(px_filename, cam_filename)
    # valid_neurons = [list(np.nonzero(good_neuron[core])[0]) for core in range(4)]
    # valid_tags_for_good_neurons = [list(np.nonzero(good_cam_tag[core])[0]) for core in range(4)]
    # valid_tags_for_all_neurons = [list(np.nonzero(np.logical_and(good_cam_tag[core], good_tag[core]))[0]) for core in
    #                               range(4)]
    # valid_cores_with_all_neurons = [core for core in range(4) if len(valid_tags_for_all_neurons[core]) >= 256]
    # print(valid_cores_with_all_neurons)
    # valid_cores_with_good_neurons = [core for core in range(4) if len(valid_tags_for_good_neurons[core]) >= 256]
    # print(valid_cores_with_good_neurons)
    #
    # px_verify_filename = "bad_pxs_verify_" + board_name + ".csv"
    # cam_verify_filename = "bad_cams_verify_" + board_name + ".csv"
    # good_verify_filename = "good_verify_" + board_name + ".csv"
    #
    # if not os.path.exists(px_verify_filename) or not os.path.exists(cam_verify_filename):
    #
    #
    #
    #     px_list_verify = []
    #     cam_list_verify = []
    #     good_list_verify = []
    #
    #     clear_srams(config=myConfig, neurons=range(256), source_cores=[], cores=range(4),
    #                 chips=range(number_of_chips), all_to_all=True, monitor_cam=random.randint(0, 3))
    #     model.apply_configuration(myConfig)
    #     time.sleep(0.1)
    #
    #
    #     for n in range(64):
    #         print(f"Trial {n}:")
    #
    #         sweep_cams(board=board, model=model, myConfig=myConfig, cores=valid_cores_with_all_neurons, neurons=[range(256)] * 4,
    #                    synapse=[[n for _ in range(256)] for __ in range(4)],
    #                    weight=[random.choices(range(4), k=256) for _ in range(4)],
    #                    tag=[random.sample(valid_tags_for_all_neurons[_], k=256) if _ in valid_cores_with_all_neurons else [0] * 256 for _ in range(4)],
    #                    cam_list=cam_list_verify, px_list=px_list_verify, good_list=good_list_verify, double=False)
    #
    #         sweep_cams(board=board, model=model, myConfig=myConfig, cores=valid_cores_with_good_neurons, neurons=valid_neurons,
    #                    synapse=[[n for _ in range(256)] for __ in range(4)],
    #                    weight=[random.choices(range(4), k=256) for _ in range(4)],
    #                    tag=[random.sample(valid_tags_for_good_neurons[_], k=256) if _ in valid_cores_with_good_neurons else [0] * 256 for _ in range(4)],
    #                    cam_list=cam_list_verify, px_list=px_list_verify, good_list=good_list_verify, double=False)
    #
    #     np.savetxt(px_verify_filename, np.asarray(px_list_verify).astype(int), fmt='%i', delimiter=",")
    #     np.savetxt(cam_verify_filename, np.asarray(cam_list_verify).astype(int), fmt='%i', delimiter=",")
    #     np.savetxt(good_verify_filename, np.asarray(good_list_verify).astype(int), fmt='%i', delimiter=",")
    #
    # write_cam_profile(good_neuron=good_neuron, good_cam_tag=good_cam_tag,
    #                   data_cams=np.concatenate([np.loadtxt(cam_filename, delimiter=",", dtype=int),
    #                        np.loadtxt(cam_verify_filename, delimiter=",", dtype=int)], axis=0),
    #                   total_num_all=[8 + (core in valid_cores_with_all_neurons) for core in range(4)],
    #                   total_num_good=[8 + (core in valid_cores_with_all_neurons) +
    #                                   (core in valid_cores_with_good_neurons) for core in range(4)])

# def test_cams_directly(board, number_of_chips, board_name="pink"):
#
#     model, myConfig = initialize(board=board, number_of_chips=number_of_chips)
#
#     all_neurons = np.ones((4, 256), dtype=bool)
#     all_tags = np.ones((4, 2048), dtype=bool)
#     all_synapses_visited = np.zeros((4, 256, 64), dtype=int)
#     all_synapses_failed = np.zeros((4, 256, 64), dtype=int)
#     cam_list = []
#     px_list = []
#
#     synapse_test = [[[random.sample(range(64), k=64) for _ in range(256)] for __ in range(4)] for ___ in range(4)]
#
#     for r in range(4):
#
#         clear_srams(config=myConfig, neurons=range(256), source_cores=[], cores=range(4),
#                     chips=range(number_of_chips), all_to_all=True, monitor_cam=r)
#         model.apply_configuration(myConfig)
#         time.sleep(1)
#
#         for n in range(64):
#
#             for c in random.sample(range(4), k=4):
#
#                 if any(all_neurons[c]):
#
#                     tag_test = random.sample(list(np.nonzero(all_tags[c])[0]), k=256)
#
#                     # print(tag_test)
#                     for core in range(4):
#                         for i in range(256):
#                             cam_exc = [Dynapse2Synapse() for _ in range(64)]
#                             if c == core:
#                                 weight = random.randint(0, 3)
#                                 all_synapses_visited[c, i, synapse_test[r][c][i][n]] += 1
#                                 cam_exc[synapse_test[r][c][i][n]].weight = [_ == weight for _ in range(4)]
#                                 cam_exc[synapse_test[r][c][i][n]].tag = tag_test[i]
#                                 cam_exc[synapse_test[r][c][i][n]].dendrite = Dendrite.ampa
#                             myConfig.chips[0].cores[core].neurons[i].synapses = cam_exc
#                     model.apply_configuration(myConfig)
#                     # time.sleep(0.1)
#
#                     input_events = []
#                     ts = get_fpga_time(board=board) + 100000
#                     for i in range(256):
#                         input_events += [AerConstructor(DestinationConstructor(tag=tag_test[i],
#                                                                                core=[_ == c for _ in range(4)]).destination,
#                                                         ts).aer,
#                                          AerConstructor(DestinationConstructor(tag=tag_test[i],
#                                                                                core=[_ == c for _ in range(4)]).destination,
#                                                         ts + 800).aer]
#                         ts += 1000
#                     send_events(board=board, events=input_events, min_delay=0)
#
#                     output_events = [[], []]
#                     get_events(board=board, extra_time=10000, output_events=output_events)
#                     # if plot:
#                     # plot_raster(output_events=output_events)
#
#                     counts = spike_count(output_events=output_events, show=True)
#                 # for c in cores:
#                     for i in range(256):
#                         synapse = synapse_test[r][c][i][n]
#                         tag = myConfig.chips[0].cores[c].neurons[i].synapses[synapse].tag
#                         if counts[c * 256 + i] == 0:
#                             all_synapses_failed[c, i, synapse] += 1
#                             cam_list += [[c, i, synapse, tag]]
#                         elif counts[c * 256 + i] > 30:
#                             all_neurons[c][i] = False
#                             all_tags[c][tag] = False
#                             px_list += [[c, i, synapse, tag]]
#
#                     print(f"Core {c}")
#                     print(f"good neurons {np.sum(all_neurons[c])}")
#                     print(f"good tags {np.sum(all_tags[c])}")
#                     print(f"visited synapses {np.sum(all_synapses_visited[c])}")
#                     print(f"bad synapses {np.sum(all_synapses_failed[c])}")
#
#     np.savetxt("bad_pxs_direct_" + board_name + ".csv", np.asarray(px_list).astype(int), fmt='%i', delimiter=",")
#     np.savetxt("bad_cams_direct_" + board_name + ".csv", np.asarray(cam_list).astype(int), fmt='%i', delimiter=",")
#
#
#
#
