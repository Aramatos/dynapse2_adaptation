from samna.dynapse2 import *


class SynapseConstructor:

    def __init__(self, tag, dendrite, weight, stp=False, precise_delay=False, mismatched_delay=False):
        self.synapse = Dynapse2Synapse()
        self.synapse.tag = tag
        self.synapse.dendrite = dendrite
        self.synapse.weight = weight
        self.synapse.stp = stp
        self.synapse.precise_delay = precise_delay
        self.synapse.mismatched_delay = mismatched_delay


class DestinationConstructor:

    def __init__(self, tag, core, x_hop=0, y_hop=0):
        self.destination = Dynapse2Destination()
        self.destination.tag = tag
        self.destination.core = core
        self.destination.x_hop = x_hop
        self.destination.y_hop = y_hop


class AerConstructor:
    def __init__(self, destination, timestamp):
        self.aer = NormalGridEvent()
        self.aer.event = destination

        try:
            self.aer.timestamp = timestamp
        except TypeError as e:
            print("Error occurred while setting timestamp:")
            print("Destination:", destination)
            print("Timestamp:", timestamp)
            print("Exception:", e)
            raise  # Re-raise the exception to propagate it further


class VirtualSpikeConstructor:

    def __init__(self, tag, core, timestamp):
        self.spikes = [AerConstructor(destination=DestinationConstructor(tag=tag, core=core).destination,
                                     timestamp=timestamp).aer,
                      AerConstructor(destination=DestinationConstructor(tag=tag, core=core, x_hop=-7).destination,
                                     timestamp=timestamp).aer]
