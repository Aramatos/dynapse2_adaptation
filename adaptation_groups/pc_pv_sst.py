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
from adaptation_lib.dynapse_setup  import *
import numpy as np
import matplotlib as mp
import datetime



board_names=["dev_board"]

def pc_pv_sst(board, profile_path, number_of_chips,neuron_config):
    # Determine the user's home directory
    home_directory = os.path.expanduser("~")
    # Path to the Documents directory
    documents_path = os.path.join(home_directory, "Documents")
    # Path to the dynapse-se2-data directory within Documents
    base_path = os.path.join(documents_path, "dynapse-se2-data")
    # Auto Save set up
    date_label = datetime.date.today().strftime('%Y-%m-%d')
    time_label = str(datetime.datetime.now().hour) + "-" + str(datetime.datetime.now().minute)
    # tname= 'trail'
    tname = "PC,PV & SST Network"
    # Constructing the necessary paths
    dir_path = os.path.join(base_path, tname, date_label)
    config_path = os.path.join(dir_path, "config")
    raster_path = os.path.join(dir_path, "plots", "rasters", time_label)
    plot_path = os.path.join(dir_path, "plots")
    # Creating the directories if they don't exist
    os.makedirs(config_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Initialization
    model = board.get_model()
    model.reset(ResetType.PowerCycle, (1 << number_of_chips) - 1)
    time.sleep(1)
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    test_config=config_handshake(neuron_config,time_label,dir_path,config_path,plot_path,date_label,raster_path,tname)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # set neuron latches
    set_latches(myConfig,model, neuron_config, number_of_chips)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Set up network
    print("Setting up network")
    network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips)
    input1 = network.add_virtual_group(size=neuron_config['nvn'])#normal input
    PC = network.add_group(chip=0, core=0, size=neuron_config['pcn'])
    PV = network.add_group(chip=0, core=1, size=neuron_config['pvn'])
    SST= network.add_group(chip=0, core=2, size=neuron_config['sstn'])
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Input Connections
    network.add_connection(source=input1, target=PC, probability=neuron_config['Input_PC'],
                      dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
    network.add_connection(source=input1, target=PV, probability=neuron_config['Input_PV'],stp=neuron_config['STD'],
                      dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
    network.add_connection(source=input1, target=SST, probability=neuron_config['Input_SST'],
                      dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
    #PC outgoing connections
    network.add_connection(source=PC, target=PC, probability=neuron_config['PC_PC'],
                         dendrite=Dendrite.nmda, weight=[False, True, False, False])
    network.add_connection(source=PC, target=PV, probability=neuron_config['PC_PV'],stp=neuron_config['STD'],
                         dendrite=Dendrite.ampa, weight=[False, True, False, False])
    network.add_connection(source=PC, target=SST, probability=neuron_config['PC_SST'],
                         dendrite=Dendrite.ampa, weight=[False, True, False, False])  
    #PV outgoig connecions
    network.add_connection(source=PV, target=PC, probability=neuron_config['PV_PC'],
                         dendrite=Dendrite.shunt, weight=[False, False, True, False])
    network.add_connection(source=PV, target=PV, probability=neuron_config['PV_PV'],
                         dendrite=Dendrite.gaba, weight=[False, False, True, False])
    network.add_connection(source=PV, target=SST, probability=neuron_config['PV_SST'],
                         dendrite=Dendrite.shunt, weight=[False, False, True, False])
    #SST outgoig connecions
    network.add_connection(source=SST, target=PC, probability=neuron_config['SST_PC'],
                         dendrite=Dendrite.gaba, weight=[False, False, False, True])
    network.add_connection(source=SST, target=PV, probability=neuron_config['SST_PV'],
                         dendrite=Dendrite.gaba, weight=[False, False, False, True])
    network.connect()
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    # Setting monitors
    set_monitors(myConfig,model,test_config,PC=PC,PV=PV,SST=SST)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    set_configs(myConfig,model,neuron_config)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    print("\nNetwork Config Done")
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Emulation run
    #Input event
    input_events=create_events(input1,neuron_config['nvn'],neuron_config,neuron_config['in_freq'])
    print("Input events created")
    output_events=run_dynapse(neuron_config,board,input_events)
    print("Simulation done")
    # /⅞save_config_axis1(test_config,myConfig,number_of_chips,tname)
    np.save(dir_path+"/pc_pv_sst"+time_label,  output_events)
    # /⅞print("time label: "+str(time_label))

    return output_events,test_config