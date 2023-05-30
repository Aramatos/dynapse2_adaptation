# stack board run with: python3 main.py ./bitfiles/Dynapse2Stack.bit 1
# stack board run with: python3 main.py -d ./bitfiles/Dynapse2DacTestboard.bit

# if self-built samna (no need if installed through pip)
# export PYTHONPATH=$PYTHONPATH:~/Documents/git/samna/build/src

import optparse

import samna

from lib.dynapse2_init import connect, dynapse2board

from test import test_neurons, test_routing, test_synapses, test_homeostasis, test_sadc
from measure import meas_syn_tau
from example import wta, perceptron, stp, stdp, stdp_mp, parietal
from adaptation import  ff_network, pc_pv_sst, pc_pv,pc_single,pv_single,ff_pc_pv,ff,sst_single,pc_pv_de
from configs.neuron_configs import test_selection

import os



def main():
    parser = optparse.OptionParser()
    parser.set_usage("Usage: test_sadc.py [options] bitfile [number_of_chips]")
    parser.add_option("-d", "--devboard", action="store_const", const="devboard", dest="device",
                      help="use first XEM7360 found together with DYNAP-SE2 DevBoard")
    parser.add_option("-s", "--stack", action="store_const", const="stack", dest="device",
                      help="use first XEM7310 found together with DYNAP-SE2 Stack board(s)")
    parser.set_defaults(device="stack")
    opts, args = parser.parse_args()


    if len(args) == 2:
        number_of_chips = int(args[1])
    else:
        number_of_chips = 1

    # receiver_endpoint = "tcp://0.0.0.0:33335"
    # sender_endpoint = "tcp://0.0.0.0:33336"
    # node_id = 1
    # interpreter_id = 2
    # samna_node = samna.SamnaNode(sender_endpoint, receiver_endpoint, node_id)

 
    deviceInfos = samna.device.get_unopened_devices()
    print(deviceInfos)
    
    global board
    board = samna.device.open_device(deviceInfos[0])
    board_names = ["dev"]
    board.reset_fpga()
    # remote = connect(opts.device, number_of_chips, samna_node, sender_endpoint, receiver_endpoint, node_id,
    #                  interpreter_id)
    
  
    board = dynapse2board(board=board, args=args)
    
    test_select=test_selection()
    #read test selection from json file test_select.json
    #import json
    #with open('test_select.json') as json_file:
        #test_select = json.load(json_file)
    #print(test_select)
    
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if test_select['FF_PC_PV']==True:
        data=ff_pc_pv.ff_pc_pv(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    if test_select['FF_Network']==True:
        data=ff_network.ff_network(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    if test_select['FF_Single_Neurons']==True:
        data=ff.ff_single_neurons(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    if test_select['PC_PV_Network']==True:
        data=pc_pv.pc_pv(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    if test_select['PC_PV_SST_Network']==True:
        data=pc_pv_sst.pc_pv_sst(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    if test_select['PC_Neuron']==True:
        data=pc_single.pc_single(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    if test_select['PV_Neuron']==True:
        data=pv_single.pv_single(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    if test_select['SST_Neuron']==True:
        data=sst_single.sst_single(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    if test_select['DE_PV_PC']==True:
        data=pc_pv_de.pc_pv_diff(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)

    
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # test_neurons.test_dc(board=board, number_of_chips=number_of_chips)
    # test_routing.test_cams_two_steps(board=board, number_of_chips=number_of_chips)
    # test_routing.test_cams_directly(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_weights(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_nmda(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_nmda(board=board, number_ochenxif_chips=number_of_chips)
    # test_synapses.test_delay(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_delay_broad_time(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_delay_time_interchangable_20_500_copy(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_conductance(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_alpha(board=board, number_of_chips=number_of_chips)
    # test_homeostasis.homeostasis(board=board, number_of_chips=number_of_chips)
    # test_homeostasis.homeostasis_sadc(board=board, number_of_chips=number_of_chips)
    # test_sadc.test_calibration(board=board, number_of_chips=number_of_chips)
    # test_sadc.test_adaptation(board=board, number_of_chips=number_of_chips)
    # test_sadc.test_stp(board=board, number_of_chips=number_of_chips)
    # meas_syn_tau.syn_tau_ampa(board=board, number_of_chips=number_of_chips)
    # meas_syn_tau.syn_tau_gaba(board=board, number_of_chips=number_of_chips)
    # meas_syn_tau.syn_tau_shunt(board=board, number_of_chips=number_of_chips)
    # wta.wta_basic(board=board, number_of_chips=number_of_chips)
    # perceptron.perceptron_xor(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    # parietal.parietal_decoding(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    # wta.wta_basic(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    # adaptation.STD(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    # FF.FF_out(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips)
    # adaptation.PC_PV_single(board=board, profile_path=os.getcwd() + "/profiles/", number_of_chips=number_of_chips,show=True)
    # stp.short_term_potentiation(board=board, number_of_chips=number_of_chips)
    # stdp.learn_to_divide(board=board, number_of_chips=number_of_chips)
    # stdp_mp.learn_to_divide(board=board, number_of_chips=number_of_chips)

    return data

if __name__ == '__main__':
    data=main()




