import os
import json
import itertools
import random
from samna.dynapse2 import Dynapse2Destination
from lib.dynapse2_obj import *
import matplotlib.pyplot as plt


class NeuronGroup:

    def __init__(self, chip, core, size, neurons,name = ""):
        self.chip = chip
        self.core = core
        self.size = size
        self.name = name
        self.neurons = neurons
        self.fixed_neurons = neurons
        self.srams = []
        self.srams_assignment = {}
        self.max_srams = 3
        self.num_cams = [0, 0]
        self.ids = []
        self.destinations = {}


class VirtualGroup:

    def __init__(self, size):
        self.size = size
        self.ids = []
        self.srams = []
        self.srams_assignment = {}
        self.max_srams = 10000  # any big number
        self.destinations = {}

    def get_ids(self):
        return self.ids

    def get_destinations(self):
        return self.destinations


class Connection:

    def __init__(self, source, target, probability, alias, dendrite, weight, repeat, stp, precise_delay,
                 mismatched_delay, matrix):
        self.source = source
        self.target = target
        self.probability = probability
        self.alias = alias
        self.dendrite = dendrite
        self.weight = weight
        self.repeat = repeat
        self.stp = stp
        self.precise_delay = precise_delay
        self.mismatched_delay = mismatched_delay
        self.matrix = matrix


class Network:

    def read_good_cam_list(self, filenames, tol):
        assert len(filenames) == self.num_chips, \
            f"{len(filenames)} cam profiles for tag space [0, 1024) for {self.num_chips} chips"
        prob_fail = []
        cam_good = []
        num_good_cams = []
        for filename in filenames:
            with open(filename) as f:
                chip = json.load(f)
                prob_fail += [chip["prob"]]
                cam_good += [[[[[cam <= tol for cam in msb] for msb in neuron] for neuron in core] for core in chip["cams"]]]
                # num_good_cams += [[[[sum(neuron) for neuron in core] for core in msb] for msb in chip["cams"]]]
                num_good_cams += [[[[sum([msb[0] == x[0] and msb[1] == x[1] for msb in neuron])
                                     for x in [(0, 0), (1, 0), (0, 1), (1, 1)]] for neuron in core] for core in cam_good[-1]]]
        return prob_fail, cam_good, num_good_cams

    def read_tag_groups(self, tol):
        tag_candidates = [[list(range(2048)),
                           list(range(512, 2048)),
                           [1032 + i * 16 + j for i in range(64) for j in range(8)],
                           list(range(1024)) + list(range(1280, 2048))]] * self.num_chips
        tag_dict = {}
        for core_combination in itertools.product(*[[True, False]] * (4 * self.num_chips)):
            tag_dict[core_combination] = []
            for tag in range(2047, -1, -1):
                if all([(tag in tag_candidates[chip][core] and self.prob_fail[chip][tag >> 10][core] < tol) ==
                        core_combination[chip * 4 + core]
                        for core in range(4) for chip in range(self.num_chips)]):
                    tag_dict[core_combination] += [tag]
        tag_groups = [tags for tags in tag_dict.values() if len(tags)]
        tag_is_good = [[[core_combination[chip * 4 + core] for core in range(4)] for chip in range(self.num_chips)]
                       for core_combination, tags in tag_dict.items() if len(tags)]
        return tag_groups, tag_is_good

    def __init__(self, config, profile_path, num_chips=1, tol=0.5 ):

        self.config = config
        self.num_chips = num_chips
        self.groups = []
        self.virtual_groups = []
        self.connections = []

        self.num_used_neurons = [[0 for core in range(4)] for chip in range(num_chips)]
        self.neuron_used = [[[False for neuron in range(256)] for core in range(4)] for neuron in range(num_chips)]

        self.prob_fail, self.cam_good, self.num_good_cams = \
            self.read_good_cam_list(filenames=[profile_path + "cams_profile_black.json"], tol=tol)
        self.tag_groups, self.tag_is_good = self.read_tag_groups(tol=tol)

        self.steps = 2

    def iter_reorder(self, start, end):
        yield start
        diff = end - start
        visited = [False] * diff
        q = 1
        for count in range(self.steps):
            if all(visited):
                return
            for p in range(1, q + 1, 2):
                i = int(diff * p / q + start + 0.5)
                i_offset = i - start - 1
                if i_offset >= 0 and not visited[i_offset]:
                    yield i
                    visited[i_offset] = True
            q *= 2
        return


    def add_group(self, chip, core, size, neurons=None,name = ""):
        group = NeuronGroup(
            chip=chip,
            core=core,
            size=size,
            neurons=neurons,
            name = name
        )
        self.groups += [group]
        return group

    def add_virtual_group(self, size):
        group = VirtualGroup(size=size)
        self.virtual_groups += [group]
        return group

    def check_neurons(self):
        for group in self.groups:
            self.num_used_neurons[group.chip][group.core] += group.size
            assert self.num_used_neurons[group.chip][group.core] <= 256, \
                f"Not enough neurons on chip {group.chip} core {group.core} for group {group}"
            if group.neurons is not None:
                assert group.size == len(group.neurons), \
                    f"Neuron ID list length {len(group.neurons)} not equal to size {group.size} for group {group}"
                for neuron in group.neurons:
                    assert not self.neuron_used[group.chip][group.core][neuron], \
                        f"Neuron ID {neuron} on chip {group.chip} core {group.core} for group {group} already in use"
                    self.neuron_used[group.chip][group.core][neuron] = True

    def remove_group(self, group):
        if group.neurons is not None:
            for neuron in group.neurons:
                self.neuron_used[group.chip][group.core][neuron] = False
        for connection in reversed(self.connections):
            if connection.source == group or connection.target == group:
                self.connections.remove(connection)
        self.groups.remove(group)

    def remove_virtual_group(self, group):
        for connection in reversed(self.connections):
            if connection.source == group:
                self.connections.remove(connection)
        self.virtual_groups.remove(group)

    def add_connection(self, source, target, probability, dendrite, weight=None,
                       repeat=1, stp=False, precise_delay=False, mismatched_delay=False, matrix=None):
        assert source in self.groups or source in self.virtual_groups, \
            f"Source group {source} undefined in either neuron groups {self.groups}" \
            f" or virtual neuron groups {self.virtual_groups}"
        assert target in self.groups, f"Target group {target} undefined in {self.groups}"
        if probability == 1 and source in self.groups:
            alias = "aliasing"
        else:
            alias = "non_aliasing"
        connection = Connection(source=source,
                                target=target,
                                probability=probability,
                                alias=alias,
                                dendrite=dendrite,
                                weight=weight,
                                repeat=repeat,
                                stp=stp,
                                precise_delay=precise_delay,
                                mismatched_delay=mismatched_delay,
                                matrix=matrix
                                )
        self.connections += [connection]
        return connection

    def remove_connection(self, source, target):
        for connection in reversed(self.connections):
            if connection.source == source and connection.target == target:
                self.connections.remove(connection)

    def check_connections(self):
        for connection in self.connections:
            if (connection.target.chip, 1 << connection.target.core, connection.alias) not in connection.source.srams:
                connection.source.srams += [(connection.target.chip, 1 << connection.target.core, connection.alias)]

    def assign_tags(self, available_tags, unassigned_neurons, tag_group, assignment):
        if tag_group == len(self.tag_groups) - 1:
            if unassigned_neurons <= available_tags[tag_group]:
                yield [a + (i == tag_group) * unassigned_neurons for i, a in enumerate(assignment)], \
                      [a - (i == tag_group) * unassigned_neurons for i, a in enumerate(available_tags)]
            return
        for n in self.iter_reorder(0, min(available_tags[tag_group], unassigned_neurons)):
            yield from self.assign_tags(
                available_tags=[a - (i == tag_group) * n for i, a in enumerate(available_tags)],
                unassigned_neurons=unassigned_neurons - n,
                tag_group=tag_group + 1,
                assignment=[a + (i == tag_group) * n for i, a in enumerate(assignment)]
            )

    def check_sram_connection(self, chip, core, assignment):
        for i in range(len(self.tag_groups)):
            if assignment[i]:
                for c in range(4):
                    if core & 1 << c > 0 and not self.tag_is_good[i][chip][c]:
                        return False
        return True

    def merge_srams(self, max_srams, srams, sram_id):
        if sram_id >= len(srams) - 1:
            if len(srams) < max_srams:
                yield srams
            return
        chip_core_alias_merge = srams[sram_id]
        for i, chip_core_alias in enumerate(srams[sram_id + 1:]):
            if chip_core_alias_merge[0] == chip_core_alias[0] and chip_core_alias_merge[2] == chip_core_alias[2]:
                merged_srams = srams[:sram_id] + [(chip_core_alias[0],
                      chip_core_alias[1] | chip_core_alias_merge[1],
                      chip_core_alias[2])] + srams[sram_id + 1:sram_id + 1 + i] + srams[sram_id + i + 2:]
                yield from self.merge_srams(
                    max_srams=max_srams,
                    srams=merged_srams,
                    sram_id=sram_id
                )
        yield from self.merge_srams(
            max_srams=max_srams,
            srams=srams,
            sram_id=sram_id + 1
        )

    def assign_sram_tags(self, group, srams_assignment, sram_id, available_tags):
        if sram_id >= len(srams_assignment):
            yield srams_assignment, available_tags
            return
        chip_core_alias = list(srams_assignment.keys())[sram_id]
        if chip_core_alias[2] == "aliasing":
            unassigned_neurons = 1
        else:
            unassigned_neurons = group.size
        for assignment, remained_tags in self.assign_tags(
                available_tags=available_tags,
                unassigned_neurons=unassigned_neurons,
                tag_group=0,
                assignment=[0] * len(self.tag_groups)
        ):
            if self.check_sram_connection(
                    chip=chip_core_alias[0],
                    core=chip_core_alias[1],
                    assignment=assignment
            ):
                srams_assignment[chip_core_alias] = assignment
                yield from self.assign_sram_tags(
                    group=group,
                    srams_assignment=srams_assignment,
                    sram_id=sram_id + 1,
                    available_tags=remained_tags
                )

    def assign_neurons(self, group_id, available_tags):
        if group_id < len(self.virtual_groups):
            group = self.virtual_groups[group_id]
        else:
            if group_id < len(self.virtual_groups) + len(self.groups):
                group = self.groups[group_id - len(self.virtual_groups)]
            else:
                return self.check_cams()
        for srams in self.merge_srams(group.max_srams, group.srams, 0):
            for srams_assignment, remained_tags in self.assign_sram_tags(
                    group=group,
                    srams_assignment={sram: [0] * len(self.tag_groups) for sram in srams},
                    sram_id=0,
                    available_tags=available_tags
            ):
                group.srams_assignment = srams_assignment
                if self.assign_neurons(
                        group_id=group_id + 1,
                        available_tags=remained_tags
                ):
                    return True
        return False

    def check_cams(self):
        for group in self.groups:
            group.neurons = group.fixed_neurons
            group.num_cams = [0, 0]
        tag_groups_msb = [(tags[0] >> 10) for tags in self.tag_groups]
        num_required_cams = [[[] for core in range(4)] for chip in range(self.num_chips)]
        requiring_group = [[[] for core in range(4)] for chip in range(self.num_chips)]
        for connection in self.connections:
            for core_combination in range(16):
                if (connection.target.chip, core_combination, connection.alias) in connection.source.srams_assignment\
                        and core_combination & 1 << connection.target.core > 0:
                    for i, num_tags in enumerate(connection.source.srams_assignment[
                                                     (connection.target.chip, core_combination, connection.alias)]):
                        connection.target.num_cams[tag_groups_msb[i]] += int(num_tags * connection.probability + 0.5) * connection.repeat
                    break
        for group in self.groups:
            if group.neurons is not None:
                for neuron in group.neurons:
                    num_good_cams_neuron = [_ for _ in self.num_good_cams[group.chip][group.core][neuron]]
                    for msb in range(2):
                        if group.num_cams[msb] > num_good_cams_neuron[1 << msb] + num_good_cams_neuron[3]:
                            return False
                        num_good_cams_neuron[1 << msb] -= min(group.num_cams[msb], num_good_cams_neuron[1 << msb])
                        num_good_cams_neuron[3] -= max(group.num_cams[msb] - num_good_cams_neuron[1 << msb], 0)
            else:
                group.neurons = []
                num_required_cams[group.chip][group.core] += [group.num_cams] * group.size
                requiring_group[group.chip][group.core] += [group] * group.size
        for chip in range(self.num_chips):
            for core in range(4):
                num_required_cams_core = sorted(
                    enumerate(num_required_cams[chip][core]), key=lambda x: sum(x[1]), reverse=True)
                num_good_cams_core = sorted(
                    [(i, n) for i, n in enumerate(self.num_good_cams[chip][core])
                     if not self.neuron_used[chip][core][i]], key=lambda x: x[1][1] + x[1][3], reverse=False)
                # plt.figure(figsize=(5, 5))
                # for num_cams in num_good_cams_core:
                #     plt.plot([num_cams[1][1] + num_cams[1][3], num_cams[1][1] + num_cams[1][3], num_cams[1][1], 0],
                #              [0, num_cams[1][2], num_cams[1][2] + num_cams[1][3], num_cams[1][2] + num_cams[1][3]])
                # plt.title("Feasible set margin")
                # plt.xlabel("#CAMs with tag 0-1023")
                # plt.ylabel("#CAMs with tag 1024-2047")
                # plt.show()
                
                for i, num_required in enumerate(num_required_cams_core):
                    flag = False
                    for j, num_good in enumerate(num_good_cams_core):
                        if num_required[1][0] <= num_good[1][1] + num_good[1][3] and \
                                num_required[1][1] <= num_good[1][2] + num_good[1][3] - \
                                max(num_required[1][0] - num_good[1][1], 0):
                            requiring_group[chip][core][num_required[0]].neurons += [num_good[0]]
                            num_good_cams_core = num_good_cams_core[:j] + num_good_cams_core[j+1:]
                            flag = True
                            break
                    if not flag:
                        return False
        # for connection in self.connections:
        #     for core_combination in range(16):
        #         if (connection.target.chip, core_combination, connection.alias) in connection.source.srams_assignment\
        #                 and core_combination & 1 << connection.target.core > 0:
        #             for i, num_tags in enumerate(connection.source.srams_assignment[
        #                                              (connection.target.chip, core_combination, connection.alias)]):
        #                 print(connection.source, connection.target, connection, core_combination, int(num_tags * connection.probability) * connection.repeat)
        #             break
        # for group in self.groups:
        #     print(group.srams_assignment)
        # print(num_required_cams)
        return True

    def config_connections(self):
        for chip in range(self.num_chips):
            for core in range(4):
                for neuron in range(256):
                    self.config.chips[chip].cores[core].neurons[neuron].destinations = [Dynapse2Destination() for _ in range(4)]
                    self.config.chips[chip].cores[core].neurons[neuron].synapses = [Dynapse2Synapse() for _ in range(64)]

        tag_ids = [0] * len(self.tag_groups)
        id_count = 0 # for monitoring
        for group in self.virtual_groups + self.groups:
            for chip_core_alias, tag_assignment in group.srams_assignment.items():
                group.destinations[(chip_core_alias[0], chip_core_alias[1])] = []
                for i, num_tags in enumerate(tag_assignment):
                    tag_id_end = tag_ids[i] + num_tags
                    group.destinations[(chip_core_alias[0], chip_core_alias[1])] +=\
                        self.tag_groups[i][tag_ids[i]:tag_id_end]
                    tag_ids[i] = tag_id_end
            group.ids = range(id_count, id_count + group.size)
            id_count += group.size + 1
        for group in self.groups:
            # print(group.destinations)
            for i, neuron in enumerate(group.neurons):
                destinations = [DestinationConstructor(tag=group.ids[i], core=[True] * 4, x_hop=-7).destination]
                for chip_core_alias, tag_assignment in group.srams_assignment.items():
                    destinations += [DestinationConstructor(
                        tag=group.destinations[
                            (chip_core_alias[0], chip_core_alias[1])][i * (chip_core_alias[2] == "non_aliasing")],
                        core=[chip_core_alias[1] & 1 << core > 0 for core in range(4)],
                        x_hop=chip_core_alias[0] - group.chip).destination]
                while len(destinations) < 4:
                    destinations += [Dynapse2Destination()]
                self.config.chips[group.chip].cores[group.core].neurons[neuron].destinations = destinations
        for group in self.groups:
            synapses = [[Dynapse2Synapse() for synapse_id in range(64)] for neuron in range(group.size)]
            group_connections = []
            for connection in self.connections:
                if connection.target == group:
                    for core_combination in range(16):
                        if (connection.target.chip, core_combination, connection.alias) in\
                                connection.source.srams_assignment and\
                                core_combination & 1 << connection.target.core > 0:
                            source_tags = connection.source.destinations[(connection.target.chip, core_combination)]
                            break
                    group_connections += [(source_tags[0] >> 10, connection, source_tags)]
            group_connections = sorted(group_connections, key=lambda x: x[0])
            for connection_source_tags in group_connections:
                connection = connection_source_tags[1]
                source_tags = connection_source_tags[2]
                # for core_combination in range(16):
                #     if (connection.target.chip, core_combination, connection.alias) in\
                #             connection.source.srams_assignment and\
                #             core_combination & 1 << connection.target.core > 0:
                #         source_tags = connection.source.destinations[(connection.target.chip, core_combination)]
                #         break
                # print(source_tags)
                for i, neuron in enumerate(group.neurons):
                    if connection.alias == "aliasing":
                        tags = [source_tags[0]] * connection.repeat
                    elif connection.matrix is not None:
                        tags = [source_tags[pre_weight[0]] for pre_weight in connection.matrix[i]]
                    else:
                        # print(int(connection.source.size * connection.probability + 0.5))
                        tags = random.sample(source_tags,
                                                k=int(connection.source.size * connection.probability + 0.5)) *\
                                connection.repeat
                    for j, tag in enumerate(tags):
                        msb = tag >> 10
                        synapse_id = 0
                        while synapse_id < 64 and\
                                (msb or not (synapses[i][synapse_id].dendrite == Dendrite.none and
                                                self.cam_good[group.chip][group.core][neuron][synapse_id][0] and
                                                not self.cam_good[group.chip][group.core][neuron][synapse_id][1])):
                            synapse_id += 1
                        if synapse_id == 64:
                            for synapse_id in range(64):
                                if synapses[i][synapse_id].dendrite == Dendrite.none and \
                                        self.cam_good[group.chip][group.core][neuron][synapse_id][msb]:
                                    break
                        # if group.core == 1 and neuron == 185:
                        #     print(synapse_id)
                        synapses[i][synapse_id].tag = tag
                        synapses[i][synapse_id].dendrite = connection.dendrite
                        if connection.matrix is not None:
                            synapses[i][synapse_id].weight = connection.matrix[i][j][1]
                        else:
                            synapses[i][synapse_id].weight = connection.weight
                        synapses[i][synapse_id].precise_delay = connection.precise_delay
                        synapses[i][synapse_id].mismatched_delay = connection.mismatched_delay
                        synapses[i][synapse_id].stp = connection.stp
            for i, neuron in enumerate(group.neurons):
                self.config.chips[group.chip].cores[group.core].neurons[neuron].synapses = synapses[i]

    def connect(self, tol=2):
        self.num_used_neurons = [[0 for core in range(4)] for chip in range(self.num_chips)]
        self.neuron_used = [[[False for neuron in range(256)] for core in range(4)] for neuron in range(self.num_chips)]
        for group in self.groups:
            group.neurons = group.fixed_neurons
            group.num_cams = [0, 0]
            group.srams = []
        self.check_neurons()
        self.check_connections()
        assert self.assign_neurons(
                group_id=0,
                available_tags=[len(tag_group) for tag_group in self.tag_groups],
        ), "Create connection failed"
        self.config_connections()
        print()


def main():
    network = Network(config=Dynapse2Configuration(), profile_path=os.getcwd() + "/../profiles/")
    # network = Network(config=None, profile_path=os.getcwd() + "/../profiles/")
    vgroup0 = network.add_virtual_group(size=10)
    group0 = network.add_group(chip=0, core=0, size=50)
    group1 = network.add_group(chip=0, core=1, size=50)
    group2 = network.add_group(chip=0, core=2, size=50)
    group3 = network.add_group(chip=0, core=3, size=50)
    # network.remove_group(group=group2)
    # network.remove_virtual_group(group=vgroup0)
    network.add_connection(source=vgroup0, target=group0, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group0, target=group0, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group1, target=group0, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group2, target=group0, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group3, target=group0, probability=1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group0, target=group1, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group1, target=group1, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group2, target=group1, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group3, target=group1, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group0, target=group2, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group1, target=group2, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group2, target=group2, probability=1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group3, target=group2, probability=1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group0, target=group3, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group1, target=group3, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group2, target=group3, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    network.add_connection(source=group3, target=group3, probability=0.1, dendrite=Dendrite.ampa, weight=[True] * 4)
    # network.remove_group(group=group2)
    # network.remove_connection(source=group2, target=group2)
    network.connect()
    # for i in iter_reorder(3, 100):
    #     print(i)


if __name__ == "__main__":
    main()
