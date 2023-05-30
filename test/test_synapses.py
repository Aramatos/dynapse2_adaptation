import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_spikegen import get_fpga_time, send_events
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *
from samna.dynapse2 import *
from lib.dynapse2_network import *


def test_weights(board, number_of_chips):

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

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 0, 50)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 1, 20)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 160)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 4, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 4, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', 4, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', 4, 120)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        for i in range(256):
            cam_exc = [Dynapse2Synapse() for _ in range(64)]
            for j in range(64):
                weights = [False, False, False, False]
                weights[j % 4] = True
                cam_exc[j].weight = weights
                if j < 8:
                    cam_exc[j].tag = 1024 + i + (j % 4) * 256
                    cam_exc[j].dendrite = Dendrite.ampa
                else:
                    cam_exc[j].tag = 0
                    cam_exc[j].dendrite = Dendrite.none
            myConfig.chips[0].cores[c].neurons[i].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set SRAM -- axons
    clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    ts = get_fpga_time(board=board) + 100000

    input_events = []
    for i in range(1024):
        input_events += [
            AerConstructor(DestinationConstructor(tag=1024+i,
                                                  core=[True, False, False, False]).destination,
                           ts + i * 10000).aer,
            AerConstructor(DestinationConstructor(tag=1024+i,
                                                  core=[True, False, False, False], x_hop=-7).destination,
                           ts + i * 10000).aer]

    print("\nAll configurations done!\n")
    send_events(board=board, events=input_events, min_delay=100000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)


def test_nmda(board, number_of_chips):
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
    tag_ampa = 1024
    tag_nmda = 1025

    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
        myConfig.chips[0].cores[c].neurons[neuron].latch_denm_nmda = True
        myConfig.chips[0].cores[c].neurons[neuron].latch_coho_ca_mem = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set dc latches
    # set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_NMREV_N', 0, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 3, 140)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 4, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        cam_exc = [Dynapse2Synapse() for _ in range(64)]
        for _ in range(4):
            cam_exc[_].weight = [True, False, False, False]
            cam_exc[_].tag = tag_ampa
            cam_exc[_].dendrite = Dendrite.ampa
        for _ in range(4, 8):
            cam_exc[_].weight = [False, True, False, False]
            cam_exc[_].tag = tag_nmda
            cam_exc[_].dendrite = Dendrite.nmda
        myConfig.chips[0].cores[c].neurons[neuron].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set SRAM -- axons
    # clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    print("\nAll configurations done!\n")

    ts = get_fpga_time(board=board) + 100000

    input_events = []

    for _ in range(1000):

        input_events += [
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 1000).aer,
            AerConstructor(DestinationConstructor(tag=tag_nmda,
                                                  core=[True, False, False, False]).destination,
                           ts + 5000).aer,
            AerConstructor(DestinationConstructor(tag=tag_nmda,
                                                  core=[True, False, False, False]).destination,
                           ts + 101000).aer,
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 105000).aer]
        ts += 300000

    send_events(board=board, events=input_events, min_delay=0)
    # output_events = [[], []]
    # get_events(board=board, extra_time=100, output_events=output_events)
    # spike_count(output_events=output_events)
    # plot_raster(output_events=output_events)

def test_delay_time_interchangable_20_500(board, number_of_chips):
    # change this, time between the inputs in ms:
    time_between_inputs_ms = 40
    pn = 10 #amount of neurons to test
    bn = 10 #start neuron

    #setup for interval and delaytime_between_inputs_ms #3*
    delay_variable = int((1/time_between_inputs_ms)*900) #happy with this
    tau_variable = int(1/(time_between_inputs_ms)*3000) #*3000 with coarse value 3
    gain_variable = 2*tau_variable
    gain_coarse = 2+(gain_variable//254)
    gain_variable = gain_variable % 254
    #overflow protection


    input_time = time_between_inputs_ms * 1000
    print(gain_variable,gain_coarse,tau_variable)

    #setup board
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    # set initial configuration
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    tag_ampa = range(1024,1280,1)
    tag_nmda = range(1280,1536,1)
    for c in range(1):
        for n in range(bn,bn+pn):
            myConfig.chips[0].cores[c].neuron_monitoring_on = True
            myConfig.chips[0].cores[c].monitored_neuron = n
            #myConfig.chips[0].cores[c].neurons[neuron].latch_denm_nmda = True
            #myConfig.chips[0].cores[c].neurons[neuron].latch_coho_ca_mem = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set dc latches
    # set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 254) #spike
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 60) #leak neuron (was 1,60) (0,80 for 140 ms) (2/140 works pretty well)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 2, 60) #refractory period
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254) #spike threshhold
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)

        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, tau_variable) # 3/12 for no buildup at 10 ms
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', gain_coarse, gain_variable)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 5, 20)  # AMPA (4/20)

        #set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 40)
        #set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 80)
        #set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_NMREV_N', 0, 60) #nmda threshold
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 5, 20) # NMDA

        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY0_P', 1, delay_variable)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY1_P', 5, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY2_P', 0, 0)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        for n in range(200,210):
            cam_exc = [Dynapse2Synapse() for _ in range(64)]
            for _ in range(1):
               cam_exc[_].weight = [True, False, False, False]
               cam_exc[_].tag = tag_ampa[n]
               cam_exc[_].dendrite = Dendrite.ampa
               cam_exc[_].precise_delay = False
            for _ in range(1):
               cam_exc[_].weight = [True, False, False, False]
               cam_exc[_].tag = tag_nmda[n]
               cam_exc[_].dendrite = Dendrite.ampa
               cam_exc[_].precise_delay = True
            myConfig.chips[0].cores[c].neurons[n].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set SRAM -- axons
    clear_srams(config=myConfig, neurons=range(256), cores=range(1), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("\nAll configurations done!\n")

    ts = get_fpga_time(board=board) + 100000

    input_events = []
    for n in range(200,201):
        for _ in range(10):
            input_events += \
                VirtualSpikeConstructor(tag=tag_nmda[n], core=[True, False, False, False], timestamp=ts).spikes + \
                VirtualSpikeConstructor(tag=tag_nmda[n], core=[True, False, False, False], timestamp=ts + input_time).spikes +\
                VirtualSpikeConstructor(tag=tag_nmda[n], core=[True, False, False, False], timestamp=ts + 2*input_time).spikes +\
                VirtualSpikeConstructor(tag=tag_nmda[n], core=[True, False, False, False], timestamp=ts + 3*input_time).spikes
            ts += 4*input_time

    send_events(board=board, events=input_events, min_delay=0)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)

def test_delay_time_interchangable_20_500_copy(board, number_of_chips):
    # change this, time between the inputs in ms:
    time_between_inputs_ms = 40
    n_a = 10 #number of neurons to test
    n_b = 200 #start neurons to test

    neurons = range(256)

    #setup for interval and delaytime_between_inputs_ms #3*
    delay_variable = int((1/time_between_inputs_ms)*700) #happy with this
    tau_variable = int(1/(time_between_inputs_ms)*3000) #*3000 with coarse value 3
    gain_variable = tau_variable
    # gain_coarse = 2+(gain_variable//254)
    # gain_variable = gain_variable % 254
    #overflow protection


    input_time = time_between_inputs_ms * 1000
    # print(gain_variable,gain_coarse,tau_variable)

    #setup board
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    # set initial configuration
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    # tag_ampa = range(1024,1280,1)
    # tag_nmda = range(1280,1536,1)
    # for c in range(1):
    #     # for n in range(1):
    #     myConfig.chips[0].cores[c].neuron_monitoring_on = True
    #     myConfig.chips[0].cores[c].monitored_neuron = 1
    #         #myConfig.chips[0].cores[c].neurons[neuron].latch_denm_nmda = True
    #         #myConfig.chips[0].cores[c].neurons[neuron].latch_coho_ca_mem = True
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # # set dc latches
    # set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 254) #spike
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 80) #leak neuron (was 1,60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 2, 60) #refractory period
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254) #spike threshhold
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)

        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 1, 100) # 3/12 for no buildup at 10 ms
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 1, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 4, 18)  # AMPA (4/20)

        #set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 40)
        #set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 80)
        #set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_NMREV_N', 0, 60) #nmda threshold
        # set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 4, 30) # NMDA

        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
        # set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY0_P', 1, delay_variable)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY0_P', 0, 130)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY1_P', 5, 200)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY2_P', 0, 0)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set CAM -- synapses
    # for c in range(1):
    #     for n in neurons:
    #         cam_exc = [Dynapse2Synapse() for _ in range(64)]
    #         for _ in range(0, 4):
    #            cam_exc[_].weight = [True, False, False, False]
    #            cam_exc[_].tag = tag_ampa[n]
    #            cam_exc[_].dendrite = Dendrite.ampa
    #            cam_exc[_].precise_delay = False
    #         for _ in range(4, 8):
    #            cam_exc[_].weight = [True, False, False, False]
    #            cam_exc[_].tag = tag_nmda[n]
    #            cam_exc[_].dendrite = Dendrite.ampa
    #            cam_exc[_].precise_delay = True
    #         myConfig.chips[0].cores[c].neurons[n].synapses = cam_exc
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)
    #
    # # set SRAM -- axons
    # clear_srams(config=myConfig, neurons=range(256), cores=range(1), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    network = Network(config=myConfig, profile_path=os.getcwd() + "/profiles/", num_chips=number_of_chips)
    groups = []
    ampa_groups = []
    nmda_groups = []
    for n in neurons:
        ampa_groups += [network.add_virtual_group(size=1)]
        nmda_groups += [network.add_virtual_group(size=1)]
        groups += [network.add_group(chip=0, core=0, size=1)]
        network.add_connection(ampa_groups[-1], groups[-1], 1,
                               Dendrite.ampa, [True, False, False, False], repeat=4, precise_delay=False)
        network.add_connection(nmda_groups[-1], groups[-1], 1,
                               Dendrite.ampa, [True, False, False, False], repeat=4, precise_delay=True)
    network.connect()
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # for group in ampa_groups:
    #     print(group.ids)
    #
    # for group in nmda_groups:
    #     print(group.ids)
    #
    # for group in groups:
    #     print(group.ids)

    tag_ampa = [_ for ampa in ampa_groups for v in ampa.get_destinations().values() for _ in v]
    tag_nmda = [_ for nmda in nmda_groups for v in nmda.get_destinations().values() for _ in v]

    for c in range(1):
        # for n in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = groups[1].neurons[0]
            #myConfig.chips[0].cores[c].neurons[neuron].latch_denm_nmda = True
            #myConfig.chips[0].cores[c].neurons[neuron].latch_coho_ca_mem = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # for group in groups:
    #     print(group.ids)
    #     for syn in myConfig.chips[0].cores[0].neurons[group.neurons[0]].synapses:
    #         print(syn.tag)
    #         print(syn.dendrite)

    # print(tag_ampa)

    print("\nAll configurations done!\n")

    for delay_fine in range(100, 150, 10):
        for weight_fine in range(100, 150, 10):
            print(f"Weight fine value {weight_fine}")
            set_parameter(myConfig.chips[0].cores[0].parameters, 'SYAM_W0_P', 3, weight_fine)
            print(f"Delay fine value {delay_fine}")
            set_parameter(myConfig.chips[0].cores[0].parameters, 'SYPD_DLY0_P', 0, delay_fine)
            model.apply_configuration(myConfig)
            time.sleep(0.1)

            ts = get_fpga_time(board=board) + 100000

            input_events = []
            for n in range(len(neurons)):
                for _ in range(1):
                    input_events += \
                        VirtualSpikeConstructor(tag=tag_ampa[n], core=[True, False, False, False], timestamp=ts).spikes + \
                        VirtualSpikeConstructor(tag=tag_nmda[n], core=[True, False, False, False], timestamp=ts + input_time).spikes +\
                        VirtualSpikeConstructor(tag=tag_nmda[n], core=[True, False, False, False], timestamp=ts + 2*input_time).spikes +\
                        VirtualSpikeConstructor(tag=tag_ampa[n], core=[True, False, False, False], timestamp=ts + 3*input_time).spikes
                    ts += 4*input_time

            send_events(board=board, events=input_events, min_delay=0)
            output_events = [[], []]
            get_events(board=board, extra_time=100, output_events=output_events)
            counts = spike_count(output_events=output_events, show=False)
            neuron_counts = [counts[i] for i in range(1024, 1536, 2)]
            print(f"No response {sum([_ == 0 for _ in neuron_counts])}, false positive {sum([_ >= 2 for _ in neuron_counts])}")
            plot_raster(output_events=output_events)

def test_conductance(board, number_of_chips):
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
    tag_ampa = 1024
    tag_cond = 1025

    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
        myConfig.chips[0].cores[c].neurons[neuron].latch_de_conductance = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set dc latches
    # set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_REV_N', 3, 70)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_REV_N', 5, 70)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 3, 140)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 3, 140)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        cam_exc = [Dynapse2Synapse() for _ in range(64)]
        for _ in range(4):
            cam_exc[_].weight = [True, False, False, False]
            cam_exc[_].tag = tag_ampa
            cam_exc[_].dendrite = Dendrite.ampa
        for _ in range(4, 8):
            cam_exc[_].weight = [False, True, False, False]
            cam_exc[_].tag = tag_cond
            cam_exc[_].dendrite = Dendrite.nmda
        myConfig.chips[0].cores[c].neurons[neuron].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set SRAM -- axons
    # clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    print("\nAll configurations done!\n")

    ts = get_fpga_time(board=board) + 100000

    input_events = []

    for _ in range(1000):

        input_events += [
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 1000).aer,
            AerConstructor(DestinationConstructor(tag=tag_cond,
                                                  core=[True, False, False, False]).destination,
                           ts + 5000).aer,
            AerConstructor(DestinationConstructor(tag=tag_cond,
                                                  core=[True, False, False, False]).destination,
                           ts + 101000).aer,
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 105000).aer]
        ts += 300000

    send_events(board=board, events=input_events, min_delay=0)
    # output_events = [[], []]
    # get_events(board=board, extra_time=100, output_events=output_events)
    # spike_count(output_events=output_events)
    # plot_raster(output_events=output_events)



def test_alpha(board, number_of_chips):
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
    tag_ampa = 1024
    tag_cond = 1025

    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
        myConfig.chips[0].cores[c].neurons[neuron].latch_deam_alpha = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set dc latches
    # set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 0, 120)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 1, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ITAU_P', 0, 140)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_IGAIN_P', 1, 100)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 3, 75)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 3, 150)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        cam_exc = [Dynapse2Synapse() for _ in range(64)]
        for _ in range(4):
            cam_exc[_].weight = [True, False, False, False]
            cam_exc[_].tag = tag_ampa
            cam_exc[_].dendrite = Dendrite.ampa
        for _ in range(4, 8):
            cam_exc[_].weight = [False, True, False, False]
            cam_exc[_].tag = tag_cond
            cam_exc[_].dendrite = Dendrite.nmda
        myConfig.chips[0].cores[c].neurons[neuron].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set SRAM -- axons
    # clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    print("\nAll configurations done!\n")

    ts = get_fpga_time(board=board) + 100000

    input_events = []

    for _ in range(1000):

        input_events += [
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts).aer,
            AerConstructor(DestinationConstructor(tag=tag_cond,
                                                  core=[True, False, False, False]).destination,
                           ts + 20000).aer,
            AerConstructor(DestinationConstructor(tag=tag_cond,
                                                  core=[True, False, False, False]).destination,
                           ts + 300000).aer,
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 320000).aer]
        ts += 700000

    send_events(board=board, events=input_events, min_delay=0)
    # output_events = [[], []]
    # get_events(board=board, extra_time=100, output_events=output_events)
    # spike_count(output_events=output_events)
    # plot_raster(output_events=output_events)