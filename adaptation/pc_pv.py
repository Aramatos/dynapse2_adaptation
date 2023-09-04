from imp import init_frozen
from pickle import TRUE
from pickletools import uint8
import time
import sys
import os


from venv import create

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_network import Network
from lib.dynapse2_spikegen import get_fpga_time, send_virtual_events, poisson_gen, isi_gen,regular_gen
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *
from adaptation_lib.spike_stats import *
from adaptation_lib.dynapse_setup import *
from configs import neuron_configs_bio
import matplotlib as mp
import datetime



board_names=["dev_board"]

def pc_pv(board, profile_path, number_of_chips):
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Auto_Save_set_up
    date_label = datetime.date.today().strftime('%Y-%m-%d')
    time_label = str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)
    #tname= 'trail'
    tname = "PC & PV Network"
    dir_path = f"./data/{tname}/{date_label}"
    config_path = f"{dir_path}/config"
    raster_path = f"{dir_path}/plots/rasters/{time_label}"

    plot_path = f"{dir_path}/plots"
    os.makedirs(f"{config_path}",exist_ok=True)
    os.makedirs(f"{plot_path}",exist_ok=True)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Initialization
    model = board.get_model()
    model.reset(ResetType.PowerCycle, (1 << number_of_chips) - 1)
    time.sleep(1)
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Set_up_parameters
    nvn=10
    pvn=10
    pcn=80
    sstn=0
    neuron_config=neuron_configs()
    in_freq=neuron_config['in_freq']
    in_DC=neuron_config['in_DC']
    duration=neuron_config['duration']
    test_config=config_handshake(neuron_config,nvn,pvn,pcn,sstn,time_label,dir_path,config_path,plot_path,date_label,raster_path,tname)

    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # set neuron latches
    set_latches(myConfig,model, neuron_config, number_of_chips)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Set up network
    network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips)
    input1 = network.add_virtual_group(size=nvn)#normal input
    PC = network.add_group(chip=0, core=0, size=pcn)
    PV = network.add_group(chip=0, core=1, size=pvn)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Input Connections
    network.add_connection(source=input1, target=PC, probability=.8,
                      dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
    network.add_connection(source=input1, target=PV, probability=.1,stp=neuron_config['STD'],
                      dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
    #PC outgoing connections
    network.add_connection(source=PC, target=PC, probability=neuron_config['PC_PC'],
                         dendrite=Dendrite.nmda, weight=[False, True, False, False])
    network.add_connection(source=PC, target=PV, probability=neuron_config['PC_PV'],stp=neuron_config['STD'],
                         dendrite=Dendrite.ampa, weight=[False, True, False, False])
    #PV outgoig connecions
    network.add_connection(source=PV, target=PC, probability=neuron_config['PV_PC'],
                         dendrite=Dendrite.shunt, weight=[False, False, True, False])
    network.add_connection(source=PV, target=PV, probability=neuron_config['PV_PV'],
                         dendrite=Dendrite.gaba, weight=[False, False, True, False])
    network.connect()
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    # Setting monitors
    set_monitors(myConfig,model,test_config,PC=PC,PV=PV,SST=0)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # PC neuron parameters
    set_configs(myConfig,model,neuron_config)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #///////////////////////////////////////////////////////////////////////////////////
    #Pre simulation data
    print("\nPC Neurons\n")
    print(PC.neurons)
    print("\nPV Neurons\n")
    print(PV.neurons)
    print("\nPV Neurons\n")
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    print("\nAll configurations done!\n")
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Emulation run
    #Input event
    input_events=create_events(input1,nvn,neuron_config,in_freq,duration)
    output_events=run_dynapse(neuron_config,board,input_events)
    [cv_values,synchrony_values]=run_dynamic_anal(output_events,test_config)
    rates=spike_count(output_events=output_events,show=False)
    pop_rates(rates,test_config)
    Network_raster_plot(test_config,output_events,neuron_config,cv_values=cv_values,syn_values=synchrony_values,save=True,show=True,annotate=True,annotate_network=True)
    frequency_over_time(test_config,output_events)
    # /⅞save_config_axis1(test_config,myConfig,number_of_chips,tname)
    np.save(dir_path+"/pc_pv_s"+time_label,  output_events)
    # /⅞print("time label: "+str(time_label))

    return 