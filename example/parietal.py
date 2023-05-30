import time
import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import set_parameter, clear_srams
from lib.dynapse2_network import Network
from lib.dynapse2_spikegen import get_fpga_time, send_virtual_events, poisson_gen, isi_gen
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *

import random

fan_in = 20

from samna.dynapse2 import *


def int_to_bool_list(num):
    return [bool(num & (1 << n)) for n in range(4)]


def read_weight_matrix(filename, class_rate):

    n_in = 124

    w = np.load(filename)
    n_mid = int(len(w)/n_in)

    w_matrix = np.reshape(w, (n_in, n_mid))
    w_indices = np.argsort(w_matrix, axis=0)

    w_pruned = []

    for i in range(n_mid):
        w_pruned += [[w_matrix[j, i] * (j in w_indices[-fan_in:, i]) / class_rate[i // (n_mid // 9)] for j in range(n_in)]]

    w_pruned = np.transpose(w_pruned)

    w_pruned_flatten = w_pruned.flatten()
    w_pruned_flatten = w_pruned_flatten[w_pruned_flatten > 0]
    w_resolution_bits = 3
    w_range = np.max(w_pruned_flatten) - np.min(w_pruned_flatten)
    w_base = [np.min(w_pruned_flatten)]
    for i in range(w_resolution_bits):
        w_base += [w_range / (2 ** i) / 2]
    w_base.reverse()
    w_quant = [[(j, int_to_bool_list(np.round(w_pruned[j, i] / w_base[0]).astype('int')))
                for j in range(n_in) if w_pruned[j, i] > 0] for i in range(n_mid)]

    return w_base, w_quant


def read_class_rate(class_trial):
    spike_t = np.load('/media/chenxi/EC16-25DC/spike_t_ep4.npy', allow_pickle=True)
    class_rate = np.zeros(9)
    for class_id, trial_id in zip(range(9), range(10)):
        class_rate[class_id] += len(spike_t[class_id][trial_id])
    return class_rate / np.mean(class_rate)


def read_input_recording(input_events, input_groups, class_trial):
    spike_t = np.load('/media/chenxi/EC16-25DC/spike_t_ep4.npy', allow_pickle=True)
    spike_i = np.load('/media/chenxi/EC16-25DC/spike_i_ep4.npy', allow_pickle=True)
    for class_id, trial_id in class_trial:
        spike_sorted = np.argsort(spike_t[class_id][trial_id])
        # input_events += [isi_gen([input_groups[spike_i[class_id][trial_id][idx]] for idx in spike_sorted],
        #                         [spike_t[class_id][trial_id][idx] * 1000 for idx in spike_sorted])]
        input_events += [isi_gen(input_groups, [spike_i[class_id][trial_id][idx] for idx in spike_sorted],
                                [spike_t[class_id][trial_id][idx] * 1000 for idx in spike_sorted])]


def parietal_decoding(board, profile_path, number_of_chips):
    n_in = 124
    n_mid = 450
    n_noise = n_mid
    trials = range(0, 10)
    class_trial = [(i, j) for j in trials for i in range(9)]

    # your code starts here
    model = board.get_model()
    model.reset(ResetType.PowerCycle, (1 << number_of_chips) - 1)
    time.sleep(1)

    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    class_rate = read_class_rate(class_trial=class_trial)
    # w_base, w_quant = read_weight_matrix(filename='example/w_plast_25_f24_20read.npy', class_rate=class_rate)
    w_base, w_quant = read_weight_matrix(filename='example/w_train_25_f24_ss.npy', class_rate=class_rate)

    # # for each core, set the neuron to monitor
    # neuron = 3
    # for c in range(1):
    #     myConfig.chips[0].cores[c].neuron_monitoring_on = True
    #     myConfig.chips[0].cores[c].monitored_neuron = neuron
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # set neuron parameters
    for c in range(4):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 0, 50)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 1, 255)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 2, 255)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set synapse parameters  -- enabled AM and SC
    weight_mismatch = [150, 100, 100, 50]
    for c in range(4):
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 1, 160)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 1, 160)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 2, 20)
        for i in range(4):
            set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W' + str(i) + '_P', 3, int(w_base[i] * weight_mismatch[c]))
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 4, 80)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips)
    # input_neurons = [network.add_virtual_group(size=1) for _ in range(n_in)]
    input_neurons = network.add_virtual_group(size=n_in)
    noise_neurons = network.add_virtual_group(size=n_noise)
    if n_mid > 256:
        # mid_neurons = [network.add_group(chip=0, core=0, size=1) for _ in range(n_mid // 2)] + \
        #               [network.add_group(chip=0, core=3, size=1) for _ in range(n_mid // 2)]
        mid_neurons = [network.add_group(chip=0, core=0, size=n_mid // 2),
                       network.add_group(chip=0, core=3, size=n_mid // 2)]
        network.add_connection(source=input_neurons, target=mid_neurons[0], probability=fan_in/n_in, dendrite=Dendrite.ampa,
                               matrix=w_quant[:n_mid // 2])
        network.add_connection(source=input_neurons, target=mid_neurons[1], probability=fan_in/n_in, dendrite=Dendrite.ampa,
                               matrix=w_quant[n_mid // 2:])
        network.add_connection(source=noise_neurons, target=mid_neurons[0],
                               probability=1 / n_noise, dendrite=Dendrite.nmda, weight=[True] * 4)
        network.add_connection(source=noise_neurons, target=mid_neurons[1],
                               probability=1 / n_noise, dendrite=Dendrite.nmda, weight=[True] * 4)
    else:
        mid_neurons = [network.add_group(chip=0, core=0, size=n_mid)]
        network.add_connection(source=input_neurons, target=mid_neurons[0], probability=fan_in/n_in,
                               dendrite=Dendrite.ampa, matrix=w_quant)
        network.add_connection(source=noise_neurons, target=mid_neurons[0],
                               probability=1 / n_noise, dendrite=Dendrite.nmda, weight=[True] * 4)

    # for post, connections in enumerate(w_quant):
    #     for pre_weight in connections:
    #         network.add_connection(source=input_neurons[pre_weight[0]], target=mid_neurons[post], probability=1,
    #                                dendrite=Dendrite.ampa, weight=pre_weight[1])
    # for post in mid_neurons:
    #     network.add_connection(source=noise_neurons, target=post,
    #                            probability=1/n_mid, dendrite=Dendrite.nmda, weight=[True]*4)
    network.connect()
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # for syn in range(64):
    #     print(myConfig.chips[0].cores[0].neurons[mid_neurons[0].ids[0]].synapses[syn].tag)
    #     print(myConfig.chips[0].cores[0].neurons[mid_neurons[0].ids[0]].synapses[syn].weight)
    # for g in range(9):
    #     print(f"group {g} neuron ids: {[mid_neurons[i].ids[0] for i in range(g * n_mid // 9, (g + 1) * n_mid // 9)]}")
    print(mid_neurons[0].ids)
    print(mid_neurons[1].ids)
    nrn_offset = mid_neurons[0].ids[0]
    print(nrn_offset)

    input_events = []
    read_input_recording(input_events=input_events, input_groups=input_neurons, class_trial=class_trial)
    for trial_events in input_events:
        trial_events += poisson_gen(start=0, duration=trial_events[-1][-1], virtual_groups=[noise_neurons], rates=[20])
        trial_events.sort(key=lambda x: x[-1])

    print("\nAll configurations done!\n")

    output_events_all = [[], []]
    counts_all = []

    for i, c_t in enumerate(class_trial):
        tau_coarse = 1
        tau_fine = 250
        too_low = None
        while True:
            for c in range(4):
                set_parameter(myConfig.chips[0].cores[c].parameters, "DEAM_ETAU_P", tau_coarse, tau_fine)
            model.apply_configuration(myConfig)
            time.sleep(0.1)
            ts = get_fpga_time(board=board) + 300000
            send_virtual_events(board=board, virtual_events=input_events[i], offset=ts, min_delay=100000)
            # print("sent all events")
            output_events = [[], []]
            get_events(board=board, extra_time=100, output_events=output_events)
            # print("Got all events")
            output_events_all[0] += output_events[0]
            output_events_all[1] += output_events[1]
            counts = spike_count(output_events=output_events, show=False)
            # print("Done counting")
            # counts = [sum(counts[nrn_offset + c * n_mid // 9 * 2:nrn_offset + (c + 1) * n_mid // 9 * 2]) for c in range(9)]
            counts = [sum(counts[nrn_offset + (c > 4) + c * n_mid // 9:nrn_offset + (c + 1) * n_mid // 9 + (c >= 4)])
                      for c in range(9)]
            print(f"Trial {c_t[1]} class {c_t[0]} tau ({tau_coarse}, {tau_fine}) counts: {counts}")
            if (too_low is None or too_low) and max(counts) < n_mid // 9 * 5:
                tau_fine -= 10
                if tau_fine == 0:
                    break
            elif (too_low is None or not too_low) and max(counts) > n_mid // 9 * 10:
                too_low = False
                tau_fine += 10
                if tau_fine > 255:
                    tau_coarse += 1
                    if tau_coarse > 5:
                        break
                    tau_fine = 30
            else:
                break
        counts_all += [counts]

    plot_raster(output_events=output_events_all)
    np.savetxt('example/parietal_events_'+ str(n_mid) + '.csv', output_events_all, delimiter=',')
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(counts_all))
    plt.show()
