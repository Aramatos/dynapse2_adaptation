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
from adaptation_lib.graphing import *
from configs.neuron_configs import neuron_configs
import numpy as np
import matplotlib as mp
import datetime

board_names=["dev_board"]

def pc_single(board, profile_path, number_of_chips,neuron_config=neuron_configs()):
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Auto_Save_set_up
    date_label = datetime.date.today().strftime('%Y-%m-%d')
    time_label = str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)
    #tname= 'trail'
    tname = "PC Neuron"
    dir_path = f"./data/{tname}/{date_label}"
    config_path = f"{dir_path}/config"
    raster_path = f"{dir_path}/plots/rasters/{time_label}"
    plot_path = f"{dir_path}/plots"
    os.makedirs(f"{config_path}",exist_ok=True)
    os.makedirs(f"{plot_path}",exist_ok=True)
    os.makedirs(f"{raster_path}",exist_ok=True)
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
    pvn=0
    pcn=200
    sstn=0
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
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Input Connections
    network.add_connection(source=input1, target=PC, probability=.8,
                      dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
    #SST outgoig connecions
    network.connect()
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    # Setting monitors
    print("Setting monitors")
    h=0
    c=0
    myConfig.chips[h].cores[c].neuron_monitoring_on = True
    myConfig.chips[h].cores[c].monitored_neuron = PC.neurons[10]  
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    set_configs(myConfig,model,neuron_config)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #///////////////////////////////////////////////////////////////////////////////////
    #Pre simulation data
    print("\nPC Neurons\n")
    print(PC.neurons)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    print("\nAll configurations done!\n")
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Emulation run
    #Input event
    if neuron_config['sweep']==True:
        sweep_run(neuron_config,myConfig,model,test_config,input1,board)
    else:
        #create input events
        input_events=create_events(input1,nvn,neuron_config,in_freq,duration)
        if neuron_config['input_type']=='DC':
            DC_input(myConfig,model,in_DC)
        #obtain output events
        output_events=run_dynapse(neuron_config,board,input_events)
        rates=spike_count(output_events=output_events,show=False)
        pop_rates(rates,test_config,show=True)
        #obtain cv and cc values
        [cv_values,synchrony_values]=run_dynamic_anal(output_events,test_config)
        #plotting
        script_annotated_raster_plot(test_config,output_events,neuron_config,cv_values=cv_values,syn_values=synchrony_values,save=True,show=True,annotate=True,annotate_network=False)
        #PSTH
        [spike_times_nvn,spike_times_pvn,spike_times_pcn,spike_times_sstn]=spike_time_arrays(output_events,nvn,pvn,pcn,sstn)
        plot_psth(test_config,spike_times_pcn, bin_size=0.025)
        #FOT
        fot_output=frequency_over_time(test_config,output_events)
        frequency_vs_time_plot(fot_output,test_config,save=False,annotate=False,show=False)
        np.save(test_config['dir_path']+"/pc_"+test_config['time_label'],  output_events)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    print("time label: "+str(time_label))
    return output_events

