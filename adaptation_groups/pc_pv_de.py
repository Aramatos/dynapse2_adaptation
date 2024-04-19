from imp import init_frozen
from pickle import TRUE
from pickletools import uint8
import pandas as pd
import time
import sys
import os

from scipy.optimize import differential_evolution
from venv import create

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_network import Network
from lib.dynapse2_spikegen import get_fpga_time, send_virtual_events, poisson_gen, isi_gen,regular_gen
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *
from adaptation_lib.spike_stats import *
from adaptation_lib.dynapse_setup import *
from dynapse2_adaptation.configs.neuron_configs_bio import neuron_configs
import numpy as np
import matplotlib as mp
import datetime



board_names=["dev_board"]


def pc_pv_diff(board, profile_path, number_of_chips):
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Auto_Save_set_up
    date_label = datetime.date.today().strftime('%Y-%m-%d')
    time_label = str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)
    #tname= 'trail'
    tname = "PC & PV Search"
    dir_path = f"./data/{tname}/{date_label}"
    config_path = f"{dir_path}/config"
    plot_path = f"{dir_path}/plots"
    raster_path = f"{dir_path}/plots/rasters/{time_label}"
    os.makedirs(f"{config_path}",exist_ok=True)
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
    pvn=10
    pcn=80
    sstn=0
    neuron_config=neuron_configs()
    in_freq=neuron_config['in_freq']
    in_DC=neuron_config['in_DC']
    duration=neuron_config['duration']
    test_config={'duration':duration,
                 'in_DC':in_DC,
                 'test_name':tname,
                 'in_freq':in_freq,
                 'nvn':nvn,'pvn':pvn,'pcn':pcn,'sstn':sstn,
                 'time_label':time_label,
                 'dir_path':dir_path,
                 'config_path':config_path,
                 'plot_path':plot_path,
                 'date_label':date_label,
                 'raster_path':raster_path,
                 'input_type':neuron_config['input_type']}
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
    network.add_connection(source=input1, target=PV, probability=.8,stp=neuron_config['STD'],
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
    drain_neurons(myConfig,model)
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
    #Emulation run
    #Input event
    search_key=1
    if search_key==1:             
      niter=neuron_config['n_iter']
      input_events=create_events(input1,nvn,neuron_config,in_freq,duration)
      random_search(test_config,neuron_config,board,model,input_events,myConfig,niter)
    elif search_key==2:
      bounds = [(0,32768),(0,32768),(0,32768),(0,32768),(0,32768),(0,32768),(0,32768),(0,32768),(0,32768),(0,32768),(0,32768)]
      result = differential_evolution(make_f_1(board,model,input_events,myConfig,nvn,pvn,pcn,sstn), bounds,maxiter=100,tol=.02)
      print(result)
      np.save(result)

      


def random_search(test_config,neuron_config,board,model,input_events,myConfig,niter):
  params_list=[]
  for i in range(niter):
    x,y=random_biases('coarse_dist')
    params=list(map(list, zip(*[x,y])))

    c=0
    set_parameter(myConfig.chips[0].cores[c].parameters,'SOIF_LEAK_N',neuron_config['PC_LEAK'][0],  neuron_config['PC_LEAK'][1]) #ampa inpit
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', x[0], y[0]) #ampa inpit
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', x[1], y[1]) #ampa recurrent
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', x[2], y[2]) #inhibtion PV
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', x[3], y[3]) #ampa gain
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', x[4],y[4]) # excitatory gain
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', x[5],y[5]) # shunt gain
    c=1
    set_parameter(myConfig.chips[0].cores[c].parameters,'SOIF_LEAK_N',neuron_config['PV_LEAK'][0],  neuron_config['PV_LEAK'][1]) #gaba gain
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', x[6], y[6]) #ampa inpit
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', x[7], y[7]) #ampa recurrent
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', x[8], y[8]) #inhibtion PV
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', x[9], y[9]) #ampa gain
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_IGAIN_P',x[10],  y[10]) #gaba gain
    model.apply_configuration(myConfig)
    time.sleep(0.3)
    output_events=run_dynapse(neuron_config,board,input_events)
    [cv_values,synchrony_values]=run_dynamic_anal(output_events,test_config)
    drain_neurons(myConfig,model)
    if neuron_config['plot_iter']==True:
      Network_raster_plot(test_config,output_events,neuron_config,cv_values,save=False,show=True,annotate=False,annotate_network=False)
    [error,cv_error,synchrony_error]=analysis_error(test_config,cv_values,synchrony_values)
    params.append(error)
    cv_values=np.round(cv_values,2)
    synchrony_values=np.round(synchrony_values,2)
    params.append(cv_values[0])
    params.append(cv_values[1])
    params.append(synchrony_values[0])
    params.append(synchrony_values[1])
    params_list.append(params)
    print(params)
    df=pd.DataFrame(params_list, columns=('PC_w0','PC_w1','PC_w2','PC_AG','PC_NG','PC_SG','PV_w0','PV_w1','PV_w2','PV_AG','PV_GG','error','cv_PC','cv_PV','syn_PC','syn_PV'))
    df.to_csv('./data/RandomSearch/random_search_p_1.csv', index=False)
  time_label = str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)
  print(time_label)

def random_biases(key):
  np.random.seed()
  x=[0]*11
  if key=='random': 
    x=np.random.randint(0,5,11)
    y=np.random.randint(10,255,11)
    return x,y
  if key=='coarse_fixed':
    x[0]=3
    x[1]=3
    x[2]=3
    x[3]=3
    x[4]=3
    x[5]=3
    x[6]=3
    x[7]=3
    x[8]=3
    x[9]=3
    x[10]=3
    y=np.random.randint(10,255,11)
    return x,y
  if key=='coarse_dist':
    x=[0]*11
    x[0]=np.random.randint(3,5)
    x[1]=np.random.randint(3,5)
    x[2]=np.random.randint(3,5)
    x[3]=np.random.randint(3,5)
    x[4]=np.random.randint(3,5)
    x[5]=np.random.randint(3,5)
    x[6]=np.random.randint(3,5)
    x[7]=np.random.randint(3,5)
    x[8]=np.random.randint(3,5)
    x[9]=np.random.randint(3,5)
    x[10]=np.random.randint(3,5)
    y=np.random.randint(15,255,11)
    return x,y


def make_f_1(board,model,input_events,myConfig,nvn,pvn,pcn,sstn):
  def f(x):
    c=0
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', bias_set(x[0])[0], bias_set(x[0])[1]) #ampa inpit
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', bias_set(x[1])[0], bias_set(x[1])[1]) #ampa recurrent
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', bias_set(x[2])[0], bias_set(x[2])[1]) #inhibtion PV
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', bias_set(x[3])[0], bias_set(x[3])[1]) #ampa gain
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', bias_set(x[4])[0],bias_set(x[4])[1]) # excitatory gain
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', bias_set(x[5])[0],bias_set(x[5])[1]) # shunt gain
    c=1
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', bias_set(x[6])[0], bias_set(x[6])[1]) #ampa inpit
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', bias_set(x[7])[0], bias_set(x[7])[1]) #ampa recurrent
    set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', bias_set(x[8])[0], bias_set(x[8])[1]) #inhibtion PV
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', bias_set(x[9])[0], bias_set(x[9])[1]) #ampa gain
    set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_IGAIN_P',bias_set(x[10])[0],  bias_set(x[10])[1]) #gaba gain
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    cv_error=function_1(board,input_events,myConfig,nvn,pvn,pcn,sstn)
    return cv_error
  return f

def function_1(board,input_events,myConfig,nvn,pvn,pcn,sstn):
  print("\ngetting fpga time\n")
  ts = get_fpga_time(board=board) + 100000
  print("\nsetting virtual neurons\n")
  send_virtual_events(board=board, virtual_events=input_events, offset=ts, min_delay=100000)
  output_events = [[], []]
  get_events(board=board, extra_time=100, output_events=output_events)
  spike_count(output_events=output_events,show=True)
  cv_values=CV_Analysis(nvn,pvn,pcn,sstn,output_events,show=True)
  cv_error=(abs(cv_values[0]-1)+abs(cv_values[1]-1))/2
  return cv_error
      
      








