from samna.dynapse2 import Dynapse2Destination
import time
from lib.dynapse2_obj import *


def set_parameter(parameters, name, coarse, fine):
    parameter = parameters[name]
    parameter.coarse_value = coarse
    parameter.fine_value = fine


def clear_srams(config, neurons, cores, chips=range(1), all_to_all=False, source_cores=None, monitor_cam=0):
    # an option to differentiate between source cores (cores that send out to other neurons on chip)
    if source_cores is None:
        source_cores = cores
    assert(not all_to_all or len(chips) <= 3)
    if all_to_all:
        assert (len(chips) <= 3)
        for h in chips:
            for c in cores:
                for n in neurons:
                    config.chips[h].cores[c].neurons[n].destinations = \
                        [Dynapse2Destination()] * monitor_cam + \
                        [DestinationConstructor(tag=c*256+n, core=[True]*4, x_hop=-7).destination] + \
                        [Dynapse2Destination()] * (3 - monitor_cam)
            for c in source_cores:
                for n in neurons:
                    config.chips[h].cores[c].neurons[n].destinations = \
                        [DestinationConstructor(tag=c*256+n, core=[True]*4, x_hop=-7).destination] + \
                        [DestinationConstructor(tag=c*256+n, core=[i in cores for i in range(4)], x_hop=t - h).destination for t in chips] + \
                        [Dynapse2Destination()] * (3 - len(chips))
    else:
        for h in chips:
            for c in cores:
                for n in neurons:
                    config.chips[h].cores[c].neurons[n].destinations = \
                        [DestinationConstructor(tag=c*256+n, core=[True]*4, x_hop=-7).destination] + \
                        [Dynapse2Destination()] * 3


def set_dc_latches(config, neurons, cores, chips=range(1)):
    for h in chips:
        for c in cores:
            for n in neurons:
                config.chips[h].cores[c].neurons[n].latch_so_dc = True


def set_type_latches(config, neurons, cores, chips=range(1)):
    for h in chips:
        for c in cores:
            for n in neurons:
                config.chips[h].cores[c].neurons[n].latch_soif_type = True

def set_adaptation_latches(config, neurons, cores, chips=range(1)):
    for h in chips:
        for c in cores:
            for n in neurons:
                config.chips[h].cores[c].neurons[n].latch_so_adaptation = True