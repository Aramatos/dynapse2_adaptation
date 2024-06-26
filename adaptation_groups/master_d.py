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
from lib.dynapse2_spikegen import get_fpga_time, send_virtual_events, poisson_gen, isi_gen
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *
from adaptation_lib.spike_stats import *
from adaptation_lib.dynapse_setup import *
from dynapse2_adaptation.configs.neuron_configs_bio import neuron_configs
import numpy as np
import matplotlib as mp
import datetime



board_names=["dev_board"]


def ff_network(board, profile_path, number_of_chips):
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   #Auto_Save_set_up
   date_label = datetime.date.today().strftime('%Y-%m-%d')
   time_label = str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)
   tname = "FI Network"
   dir_path = f"./data/{tname}/{date_label}"
   config_path = f"{dir_path}/config"
   plot_path = f"{dir_path}/plots"
   raster_path = f"{dir_path}/plots/rasters/{time_label}"
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
   #read json file
   with open('test_configs.json') as f:
       test_config = json.load(f)
   with open('neuron_configs.json') as f:
       neuron_config = json.load(f)
   test_name=test_config['test_name']
   c1=test_config['c1']
   c2=test_config['c2']
   c3=test_config['c3']
   c4=test_config['c3']
   neuron_config=neuron_configs()
   in_freq=neuron_config['in_freq']
   in_DC=neuron_config['in_DC']
   test_config={'in_DC':in_DC,'tname':tname,'in_freq':in_freq,'nvn':nvn,'pvn':pvn,'pcn':pcn,'sstn':sstn,'time':time_label,'config_path':config_path,'plot_path':plot_path,'date_label':date_label,'raster_path':raster_path}
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   # set neuron latches
   # set neuron latches
   set_latches(myConfig,model, neuron_config, number_of_chips)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   # Set up network
   network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips)
   input1 = network.add_virtual_group(size=nvn)#normal input
   PC = network.add_group(chip=0, core=0, size=pcn)
   PV = network.add_group(chip=0, core=1, size=pvn)
   SST= network.add_group(chip=0, core=2, size=sstn,neurons=np.linspace(10,21,10,dtype='uint8'))
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

   model.apply_configuration(myConfig)
   time.sleep(0.1)
   #Setting monitors
   set_monitors(myConfig,model,test_config,PC=PC,PV=PV,SST=SST)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   set_configs(myConfig,model,neuron_config)
   #///////////////////////////////////////////////////////////////////////////////////
   print("\nPC Neurons\n")
   print(PC.neurons)
   print("\nPV Neurons\n")
   print(PV.neurons)
   print("\nPV Neurons\n")
   print(SST.neurons)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   print("\nAll configurations done!\n")
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   #Emulation run
   [FF_output,ovetrake_log]=FF_run(test_config,board,neuron_config,model,myConfig,input1)
   np.save(dir_path+"/ff_"+time_label, FF_output)
   np.save(dir_path+"/overtake_"+time_label, ovetrake_log)
   overtake_graph(FF_output[0],ovetrake_log,neuron_config,test_config)
   ff_graph(FF_output,test_config,neuron_config,annotate=True,annotate_network=True)

def connection_matrix(network,neuron_config):
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
