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
from adaptation_lib.graphing import *
from configs.neuron_configs import neuron_configs
import numpy as np
import matplotlib as mp
import datetime



board_names=["dev_board"]


def ff_single_neurons(board, profile_path, number_of_chips,neuron_config=neuron_configs()):
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   #Auto_Save_set_up
   date_label = datetime.date.today().strftime('%Y-%m-%d')
   time_label = str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)
   tname = "FI Single Neurons"
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
   nvn=10
   pvn=200
   pcn=200
   sstn=200
   test_config=config_handshake(neuron_config,nvn,pvn,pcn,sstn,time_label,dir_path,config_path,plot_path,date_label,raster_path,tname)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   ## set neuron latches
   set_latches(myConfig,model, neuron_config, number_of_chips)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   # Set up network
   network = Network(config=myConfig, profile_path=profile_path, num_chips=number_of_chips)
   input1 = network.add_virtual_group(size=nvn)#normal input
   PC = network.add_group(chip=0, core=0, size=pcn)
   PV = network.add_group(chip=0, core=1, size=pvn)
   SST= network.add_group(chip=0, core=2, size=sstn)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   #Input Connections
   if pcn>0:
      network.add_connection(source=input1, target=PC, probability=1,
                     dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
   if pvn>0:
      network.add_connection(source=input1, target=PV, probability=1,stp=neuron_config['STD'],
                     dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
   if sstn>0:
      network.add_connection(source=input1, target=SST, probability=1,
                     dendrite=Dendrite.ampa, weight=[True, False, False, False],repeat=1)
   network.connect()
   model.apply_configuration(myConfig)

   time.sleep(0.1)
   #Setting monitors

   # set_monitors(myConfig,model,test_config,PC,PV,SST)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   neuron_config['PC_W0']=[3,200]
   neuron_config['PV_W0']=[3,200]
   neuron_config['SST_W0']=[3,200]
   set_configs(myConfig,model,neuron_config)
   #///////////////////////////////////////////////////////////////////////////////////
   print("\nPC Neurons\n")
   print(PC.neurons)
   print("\nPV Neurons\n")
   print(PV.neurons)
   print("\nSST Neurons\n")
   print(SST.neurons)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   print("\nAll configurations done!\n")
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   if neuron_config['sweep']==True:
      sweep_variable=neuron_config['sweep_variable']
      sweep_coarse_val=neuron_config['sweep_coarse_val']
      print(f"Sweeping {sweep_variable} from {neuron_config['sweep_range_fine'][0]} to {neuron_config['sweep_range_fine'][-1]} in {neuron_config['sweep_range_fine'].size} steps")
      FF_output_array=[]
      for i in neuron_config['sweep_range_fine']:
         neuron_config[sweep_variable]=[sweep_coarse_val,int(i)]
         print(f"Sweeping {sweep_variable} to {neuron_config[sweep_variable]}")
         test_config['raster_title']='Sweep: '+sweep_variable+' '+str([sweep_coarse_val,int(i)])
         set_configs(myConfig,model,neuron_config)
         [FF_output,x]=FF_run(test_config,board,neuron_config,model,myConfig,input1)
         FF_output_array.append(FF_output)
      sweep_frequency_vs_input_plot(FF_output_array, test_config)
      np.save(dir_path+"/ff_sweep"+time_label, FF_output_array)
   else:
      [FF_output,x]=FF_run(test_config,board,neuron_config,model,myConfig,input1)
      #np.save(dir_path+"/ff_"+time_label, FF_output)
      #frequency_vs_input_plot(FF_output, test_config,neuron_config, annotate=False)
   return FF_output  