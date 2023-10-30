import time
from numpy import random
from samna.dynapse2 import *
from lib.dynapse2_obj import *
import numpy as np


def get_fpga_time(board):

    while True:

        board.input_interface_write_events(0,
        [AerConstructor(
        DestinationConstructor(tag=1024,
        core=[True]*4, x_hop=-1,
        y_hop=-1).destination,0).aer])

        # board.grid_bus_write_events(
        
        # [AerConstructor(
        
        # DestinationConstructor(tag=1024,

        # core=[True] * 4, x_hop=-1, y_hop=-1).destination,

        # 0).aer])

        for timeout in range(1000):
            evs = board.read_events()
        if len(evs) > 0:
            return evs[-1].timestamp


def send_events(board, events, min_delay=0):
    if len(events) > 0:
        ts = events[-1].timestamp
    else:
        ts = get_fpga_time(board=board)
    # board.input_interface_write_events(0, events + [AerConstructor(DestinationConstructor(tag=2047,
    #                                                                    core=[True]*4, x_hop=-1, y_hop=-1).destination,
    #                                             ts + min_delay).aer]*32)

    # can also send through the grid bus directly
    board.grid_bus_write_events(events + [AerConstructor(DestinationConstructor(tag=2047,
                                                                       core=[True]*4, x_hop=-1, y_hop=-1).destination,
                                                ts + min_delay).aer]*32)


def send_virtual_events(board, virtual_events, offset=0, min_delay=0):
    input_events = []
    for event in virtual_events:
        input_events += [AerConstructor(
            destination=DestinationConstructor(tag=event[0], core=[True]*4, x_hop=-7).destination,
            timestamp=event[3] + offset).aer] + \
                        [AerConstructor(
                            destination=DestinationConstructor(
                                tag=event[1], core=[event[2][1] & 1 << core > 0 for core in range(4)],
                                x_hop=event[2][0]).destination,
                            timestamp=event[3] + offset).aer]
      
    if len(virtual_events) > 0:
        ts = input_events[-1].timestamp
    else:
        ts = get_fpga_time(board=board)
    input_events += [AerConstructor(
        destination=DestinationConstructor(tag=2047, core=[True]*4, x_hop=-1, y_hop=-1).destination,
        timestamp=ts + min_delay).aer] * 32
    board.input_interface_write_events(0, input_events)


def poisson_gen(start, duration, virtual_groups, rates):
    events = []
    rates_weighted = [rate * group.size for group, rate in zip(virtual_groups, rates)]
    rates_weighted_sum = sum(rates_weighted)
    random.seed(51015)
    if rates_weighted_sum:
        scale = 1 / rates_weighted_sum
        p = [rate * scale for rate in rates_weighted]
        scale *= 1e6
        t = start
        while t < start + duration:
            t += random.exponential(scale=scale)
            group = random.choice(virtual_groups, p=p)
            i = random.randint(group.size)
            ids = group.get_ids()
            for chip_core, tags in group.get_destinations().items():
                events += [(ids[i], tags[i], chip_core, int(t))]
    # for group, rate in zip(virtual_groups, rates):
    #     srams = group.get_srams()
    #     for idx, tag in zip(group.get_ids(), group.get_tags()):
    #         for n in range(random.poisson(rate * duration * 1e-6)):
    #             events += [(idx, tag, srams, random.randint(start, start + duration))]
    # events.sort(key=lambda x: x[3])
    return events

def isi_gen(virtual_group, neurons, timestamps):
    events = []
    ids = virtual_group.get_ids()
    for n, t in zip(neurons, timestamps):
        for chip_core, tags in virtual_group.get_destinations().items():
            events += [(ids[n], tags[n], chip_core, int(t))]
    return events

def regular_gen(virtual_group,nvn,rate,input_duration):
    input_duration=input_duration*1e6
    if input_duration >= 1e6 :
        rate_ms=rate/1e6
        events=int(rate_ms*input_duration)
        id_list=[i for i in range(nvn)]*events
        times_stamps=list(np.linspace(0, input_duration, events))
        times_stamps=[i for i in times_stamps for j in range(nvn)]
        events=isi_gen(virtual_group,id_list,times_stamps)
    else:
        print('you have irked the code, rethink your choices')
        pass
    return events

def striated_gen(virtual_group,neuron_config,rate):
    input_type=neuron_config['input_type']
    rest_time=neuron_config['rest_time']
    duration1=neuron_config['duration1']
    duration2=neuron_config['duration2']
    if input_type=='Regular':
        nvn=10
        rate_ms=rate/1e6
        events=int(rate_ms*(duration1))
        id_list=[i for i in range(nvn)]*2*events
        print(len(id_list))
        times_stamps_1=list(np.linspace(0, duration1, events))
        times_stamps_1=[i for i in times_stamps_1 for j in range(nvn)]
        times_stamps_2=list(np.linspace(duration1+rest_time,duration2+rest_time+duration1,events))
        times_stamps_2=[i for i in times_stamps_2 for j in range(nvn)]
        times_stamps=np.asanyarray(times_stamps_1+times_stamps_2)
        print(len(times_stamps))
        events=isi_gen(virtual_group,id_list,times_stamps)
        type(isi_gen)
    elif input_type=='Poisson':
        print('HELOO')
        events=poisson_gen(0,duration1,[virtual_group],[rate])
        events=events+poisson_gen(duration1+rest_time,duration2,[virtual_group],[rate])
    elif input_type=='Poisson 2':
        pulse_duration=300000
        rest=100000
        events=poisson_gen(0,pulse_duration,[virtual_group],[100])
        events=events+poisson_gen(pulse_duration+rest,pulse_duration,[virtual_group],[200])
        events=events+poisson_gen(2*pulse_duration+2*rest,pulse_duration,[virtual_group],[100])
        events=events+poisson_gen(3*pulse_duration+3*rest,pulse_duration,[virtual_group],[300])
        events=events+poisson_gen(4*pulse_duration+4*rest,pulse_duration,[virtual_group],[500])
    elif input_type=='Poisson 3':
        pulse_duration=50000#duration of each pulse
        rest=50000 #duration of each rest
        events=poisson_gen(0,pulse_duration,[virtual_group],[20])
        events=events+poisson_gen(pulse_duration+rest,pulse_duration,[virtual_group],[20])
        events=events+poisson_gen(2*pulse_duration+2*rest,pulse_duration,[virtual_group],[20])
        events=events+poisson_gen(3*pulse_duration+3*rest,pulse_duration,[virtual_group],[20])
        events=events+poisson_gen(4*pulse_duration+4*rest,pulse_duration,[virtual_group],[20])
        events=events+poisson_gen(5*pulse_duration+5*rest,pulse_duration,[virtual_group],[20])
    

     
    else:
        print('you have irked the code, rethink your choices')
        pass
    return events

    


