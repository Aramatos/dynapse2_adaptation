import matplotlib.pyplot as plt
import numpy as np


def get_events(board, extra_time, output_events):
    # receive events
    received_last_input = False
    time_after_received_last_input = 0
    while time_after_received_last_input < extra_time:
        for ev in board.read_events():
            tag = ev.event.tag
            if ev.event.y_hop == 0:
                neuron_id = tag + (ev.event.x_hop + 6) * 2048
                output_events[0] += [neuron_id]
                output_events[1] += [ev.timestamp * 1e-6]
            elif tag == 2047:
                received_last_input = True
        if received_last_input:
            time_after_received_last_input += 1


def spike_count(output_events, show=False, jitter=False):

    if jitter:
        # if there are multiple spikes when the neuron fires once, happens for the classic ExpIF neuron type
        # which is actually a bug and will be very bad for network performance,
        # because the post neurons will always receive multiple spikes thus generate multiple PSCs
        last_t = np.zeros(2048)
        rate = np.zeros(2048)
        for i in range(len(output_events[0])):
            if last_t[output_events[0][i]] + 2e-6 < output_events[1][i]:
                rate[output_events[0][i]] += 1
            last_t[output_events[0][i]] = output_events[1][i]
    else:
        rate = np.bincount(output_events[0])
        # print(rate)
        rate = np.append(rate, [0] * (2047 - (len(rate) - 1) % 2048))

    if show and len(rate) > 0:
        for j in range(len(rate) >> 8):
            if any(rate[j * 256: (j + 1) * 256]):
                if j % 8 < 4:
                    print("Chip " + str(j >> 3) + " core " + str(j % 4) + ":")
                else:
                    print("Chip " + str(j >> 3) + " virtual core " + str(j % 4) + ":")
                for i in range(j * 256, (j + 1) * 256, 16):
                    print(str([int(rate[_]) for _ in range(i, i + 16)]) + ",")
    return rate


# def plot_raster(output_events):
#     # raster plot
#     raster = [[] for _ in range(max(output_events[0]) + 1)]
#     for i in range(len(output_events[0])):
#         raster[output_events[0][i]] += [output_events[1][i]]
#     plt.figure(figsize=(10, 5))
#     i = 0
#     gap = 0
#     for k in range(len(raster)):
#         if len(raster[k]) > 0:
#             plt.plot(raster[k], [i] * len(raster[k]), 'o', markersize=2)
#             i += 1
#             gap = False
#         else:
#             if not gap:
#                 i += 1
#             gap = True
#     plt.grid(True)
#     plt.ylabel('Neuron ID')
#     plt.show()


def plot_raster(output_events):
    # raster plot
    raster = [[] for _ in range(max(output_events[0]) + 1)]
    for i in range(len(output_events[0])):
        raster[output_events[0][i]] += [output_events[1][i]]
    plt.figure(figsize=(10, 5))
    for k in range(len(raster)):
        plt.plot(raster[k], [k] * len(raster[k]), 'o', markersize=2)
    plt.grid(True)
    plt.ylabel('Neuron ID')
    plt.show()
