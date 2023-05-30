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
from configs.neuron_configs import neuron_configs

import numpy as np
import matplotlib as mp
import datetime



board_names=["dev_board"]


def pv_single_sweep(board, profile_path, number_of_chips):
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Auto_Save_set_up
    date_label = datetime.date.today().strftime('%Y-%m-%d')
    time_label = str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)
    #tname= 'trail'
    tname = "PV Neuron Sweep"
    dir_path = f"./data/{tname}/{date_label}"
    config_path = f"{dir_path}/config"
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
    nvn=1
    pvn=200
    pcn=0
    sstn=0
    neuron_config=neuron_configs()
    in_freq=neuron_config['in_freq']
    in_DC=neuron_config['in_DC']
    test_config={'in_DC':in_DC,'tname':tname,'in_freq':in_freq,'nvn':nvn,'pvn':pvn,'pcn':pcn,'sstn':sstn,'time':time_label,'config_path':config_path,'plot_path':plot_path,'date_label':date_label}
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # set neuron latches
    set_latches(myConfig,model, neuron_config, number_of_chips)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Set up network
    network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips)
    input1 = network.add_virtual_group(size=nvn)#normal input
    PV = network.add_group(chip=0, core=1, size=pvn)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Input Connections
    network.add_connection(source=input1, target=PV, probability=1,stp=neuron_config['STD'],
                      dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
    network.connect()
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    # Setting monitors
    set_monitors(myConfig,model,test_config,PC=0,PV=PV,SST=0)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    drain_neurons(myConfig,model)
    neuron_config['PV_W0']=[3,200]
    set_configs(myConfig,model,neuron_config)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #///////////////////////////////////////////////////////////////////////////////////
    #Pre simulation data
    print("\nPV Neurons\n")
    print(PV.neurons)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    print("\nAll configurations done!\n")
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Emulation run
    #Input event
    input_events=create_events(input1,neuron_config,in_freq,neuron_config['duration'])
    if neuron_config['input_type']=='DC':
        DC_input(myConfig,model,in_DC)

    sweep_list=np.linspace(0,100,20)
    for i in sweep_list:
        output_events=run_dynapse(neuron_config,board,input_events)
        [cv_values,synchrony_values]=run_dynamic_anal(output_events,test_config)
        Network_raster_plot(test_config,output_events,neuron_config,cv_values=cv_values,syn_values=synchrony_values,save=True,show=True,annotate=True,annotate_network=False)

    rates=spike_count(output_events=output_events,show=False)
    rate_print(rates,test_config)
    Network_raster_plot(test_config,output_events,neuron_config,cv_values=cv_values,syn_values=synchrony_values,save=True,show=True,annotate=True,annotate_network=False)
    frequency_over_time(test_config,output_events,save=True,show=True)
    #Synchrony_Analsis(nvn,pcn,pvn,sstn,output_events,analysis_time,show=True)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Data aquisitons and plots
    #Plot the raster and also save the graphs
    # Save data
    np.save(dir_path+"/pv_"+time_label,  output_events)
    print("time label: "+str(time_label))
    return output_events