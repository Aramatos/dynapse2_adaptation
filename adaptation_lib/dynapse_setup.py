from cgi import test
from pickle import TRUE
import numpy as np
import warnings
import threading
import samna

from lib.dynapse2_init import connect, dynapse2board
from adaptation_lib.spike_stats import *
from adaptation_lib.graphing import *
from lib.dynapse2_util import *
from lib.dynapse2_network import Network
from lib.dynapse2_spikegen import get_fpga_time, send_virtual_events, poisson_gen,regular_gen,striated_gen
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *

def obtain_board():
    args=['./bitfiles/Dynapse2Stack.bit', '1']


    if len(args) == 2:
        number_of_chips = int(args[1])
    else:
        number_of_chips = 1

    deviceInfos = samna.device.get_unopened_devices()
    print(deviceInfos)
    
    global board
    board = samna.device.open_device(deviceInfos[0])
    board_names = ["dev"]
    board.reset_fpga()

    profile_path=os.getcwd() + "/profiles/"
    board = dynapse2board(board=board, args=args)
    
    return board,profile_path,number_of_chips


def create_events(input1,nvn,neuron_config,in_Freq):
    if neuron_config['Striated']==True:
      input_events = striated_gen(input1,neuron_config,in_Freq)
    elif neuron_config['input_type']=='Regular':
      input_events=regular_gen(input1,nvn,in_Freq,neuron_config['duration'])
    elif neuron_config['input_type']=='Poisson':
      input_events = poisson_gen(start=0, duration=neuron_config['duration']*1e6, virtual_groups=[input1], rates=[in_Freq])
    elif neuron_config['input_type']=='DC':
      input_events =[]
    else:
        warnings.warn("Input type not defined")
        pass
    return input_events

def drain_neurons(myConfig,model):
    set_parameter(myConfig.chips[0].cores[0].parameters,'SOIF_LEAK_N',5,255) #ampa inpit
    set_parameter(myConfig.chips[0].cores[1].parameters,'SOIF_LEAK_N',5,255) #gaba gain
    set_parameter(myConfig.chips[0].cores[2].parameters,'SOIF_LEAK_N',5,255) #gaba gain
    model.apply_configuration(myConfig)
    time.sleep(0.2)

def undrain_neurons(myConfig,model,neuron_config):
    set_parameter(myConfig.chips[0].cores[0].parameters,'SOIF_LEAK_N',neuron_config['PC_LEAK'][0],  neuron_config['PC_LEAK'][1]) #ampa inpit
    set_parameter(myConfig.chips[0].cores[1].parameters,'SOIF_LEAK_N',neuron_config['PV_LEAK'][0],  neuron_config['PV_LEAK'][1]) #gaba gain
    set_parameter(myConfig.chips[0].cores[2].parameters,'SOIF_LEAK_N',neuron_config['SST_LEAK'][0],  neuron_config['SST_LEAK'][1]) #gaba gain
    model.apply_configuration(myConfig)
    time.sleep(0.2)

def DC_input(myConfig,model,in_DC):
    set_parameter(myConfig.chips[0].cores[0].parameters, "SOIF_DC_P", 3, int(in_DC))
    set_parameter(myConfig.chips[0].cores[1].parameters, "SOIF_DC_P", 3, int(in_DC))
    set_parameter(myConfig.chips[0].cores[2].parameters, "SOIF_DC_P", 3, int(in_DC))
    set_parameter(myConfig.chips[0].cores[3].parameters, "SOIF_DC_P", 3, int(in_DC))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

def set_DC_parameter(myConfig,model,coarse,fine):
    set_parameter(myConfig.chips[0].cores[0].parameters, "SOIF_DC_P", int(coarse), int(fine))
    set_parameter(myConfig.chips[0].cores[1].parameters, "SOIF_DC_P", int(coarse), int(fine))
    set_parameter(myConfig.chips[0].cores[2].parameters, "SOIF_DC_P", int(coarse), int(fine))
    set_parameter(myConfig.chips[0].cores[3].parameters, "SOIF_DC_P", int(coarse), int(fine))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    return
def set_monitors(myConfig,model,test_config,PC=0,PV=0,SST=0):
   pcn=test_config['pcn']
   pvn=test_config['pvn']
   sstn=test_config['sstn']
   h=0
   print(pvn,pcn,sstn)
   if pcn>0:  
    c=0
    myConfig.chips[h].cores[c].neuron_monitoring_on = True
    myConfig.chips[h].cores[c].monitored_neuron = 10
   if pvn>0: 
    c=1
    myConfig.chips[h].cores[c].neuron_monitoring_on = True
    myConfig.chips[h].cores[c].monitored_neuron = 23  
   if sstn>0: 
    c=2
    myConfig.chips[h].cores[c].neuron_monitoring_on = True
    myConfig.chips[h].cores[c].monitored_neuron =30
   model.apply_configuration(myConfig)
   time.sleep(0.1)

def set_latches(myConfig,model, neuron_config, number_of_chips):
    if neuron_config['PC_Adaptation']==True:
        set_adaptation_latches(config=myConfig, neurons=range(256), cores=[0], chips=range(number_of_chips))
        model.apply_configuration(myConfig)
        time.sleep(1)
    if neuron_config['DC_Latches']:
        set_dc_latches(config=myConfig, neurons=range(256), cores=[0,1,2], chips=range(number_of_chips))
        model.apply_configuration(myConfig)
        time.sleep(1)   
    
def get_connection_matrix(group_size, target_count, outdegree):
    conn_matrix = []
    rng = np.random.default_rng()
    targets = rng.choice(target_count, outdegree,replace=False,shuffle=False)
    for _ in range(target_count):
        if _ in targets: 
            weights = [True, False, False, False]
        else:
            weights = [False, False, False, False]
        source_info = []
        for sindex in range(group_size):
            source_info += [(sindex,weights)]
        conn_matrix.append(source_info)
    return conn_matrix

def run_dynapse(neuron_config,board,input_events):
    print('initilize DynapSE Run')
    output_events = [[], []]
    send_virtual_events(board=board, virtual_events=[],min_delay=10000)
    get_events(board=board, extra_time=100, output_events=output_events)
    if neuron_config['input_type']=='DC':
      min_delay=neuron_config['duration']*1e6
    elif neuron_config['input_type']=='Striated':
      min_delay=neuron_config['duration']*1e6
    else:
      min_delay=10000
    print("\ngetting fpga time")
    ts = get_fpga_time(board=board) + 100000
    while True:
        ts = get_fpga_time(board=board)  # get current time of FPGA

        if ts < 2**31:
            break  # Exit the loop if ts is within the allowed range
        
        time.sleep(20)  # Pause for 20 seconds
        print(f'new ts value: {ts}')    

    # Assuming you wanted to add 0.2 to ts; make sure to convert ts to float if it's not
    ts = float(ts) + 0.2e6
    send_virtual_events(board=board, virtual_events=input_events, offset=int(ts), min_delay=int(min_delay))
    output_events = [[], []]
    get_events(board=board, extra_time=10000, output_events=output_events)
    time.sleep(1)
    return output_events


def run_dynapse_thread(neuron_config,board,input_events,result):
    print('initilize run dynapse')
    output_events = [[], []]
    send_virtual_events(board=board, virtual_events=[],min_delay=10000)
    print('initilize run dynapse')
    get_events(board=board, extra_time=100, output_events=output_events)
    if neuron_config['input_type']=='DC':
      min_delay=neuron_config['duration']*1e6
    elif neuron_config['input_type']=='Striated':
      min_delay=neuron_config['duration']*1e6
    else:
      min_delay=10000
    print("\ngetting fpga time\n")
    ts = get_fpga_time(board=board) + 100000
    print("\nsetting virtual neurons\n")
    send_virtual_events(board=board, virtual_events=input_events, offset=int(ts), min_delay=int(min_delay))
    output_events = [[], []]
    get_events(board=board, extra_time=10000, output_events=output_events)
    time.sleep(1)
    result['output'] = output_events
 

def FF_run(test_config,board,neuron_config,model,myConfig,input1):
    nvn=test_config['nvn']
    pcn=test_config['pcn']
    pvn=test_config['pvn']
    sstn=test_config['sstn']
    if neuron_config['overtake_test']==True:
        overtake_trial_log=[]
        iterations=10
    else:
        iterations=1
        overtake_trial_log=[]

    for i in range(iterations):
        if neuron_config['input_type']=='DC':
            FF_in=neuron_config['DC_FI_Range']
        else:
            FF_in=neuron_config['Freq_FI_Range']
        FF_out_PC=[]
        FF_out_PV=[]
        FF_out_SST=[]
        CV_ff_log=[]
        overtake_log=[]
        for in_Freq in FF_in:
            [rates,cv_values,synchrony_values,overtake]=FF_single_iteration(model,board,myConfig,test_config,neuron_config,input1,in_Freq)
            if len(rates)==0:
                FF_out_PC.append([0]*pcn)
                FF_out_PV.append([0]*pvn)
                FF_out_SST.append([0]*sstn) 
                CV_ff_log.append([0,0,0])
                overtake_log.append(0) 

            else:
                FF_out_PC.append(rates[nvn+1:pcn+nvn+1])
                FF_out_PV.append(rates[pcn+nvn+1:pcn+pvn+nvn+2])    
                FF_out_SST.append(rates[pcn+pvn+nvn+2:pcn+pvn+sstn+nvn+2])  
                CV_ff_log.append(cv_values)
                overtake_log.append(overtake)

        FF_out_PC=list(map(list, zip(*FF_out_PC)))
        FF_out_PV=list(map(list, zip(*FF_out_PV)))
        FF_out_SST=list(map(list, zip(*FF_out_SST)))
        FF_cv=np.round(np.mean(CV_ff_log,axis=0),2)
        FF_output=[FF_in,FF_out_PC,FF_out_PV,FF_out_SST,FF_cv]
        FF_output=np.asanyarray(FF_output,dtype=object)
        overtake_trial_log.append(overtake_log)

    return FF_output,overtake_trial_log

def FF_single_iteration(model,board,myConfig,test_config,neuron_config,input1,in_Freq):
    nvn=test_config['nvn']
    pcn=test_config['pcn']
    pvn=test_config['pvn']
    sstn=test_config['sstn']
    duration = int(neuron_config['duration'])
    test_config['in_freq']=int(in_Freq)
    test_config['in_DC']=int(in_Freq)
    output_events = [[], []]
    send_virtual_events(board=board, virtual_events=[])
    get_events(board=board, extra_time=100, output_events=output_events)
    undrain_neurons(myConfig,model,neuron_config)
    input_events=create_events(input1,nvn,neuron_config,in_Freq)
    coarse=neuron_config['DC_Coarse']
    if neuron_config['input_type']=='DC':
        set_parameter(myConfig.chips[0].cores[0].parameters, "SOIF_DC_P", coarse, int(in_Freq))
        set_parameter(myConfig.chips[0].cores[1].parameters, "SOIF_DC_P", coarse, int(in_Freq))
        set_parameter(myConfig.chips[0].cores[2].parameters, "SOIF_DC_P", coarse, int(in_Freq))
        set_parameter(myConfig.chips[0].cores[3].parameters, "SOIF_DC_P", coarse, int(in_Freq))
        model.apply_configuration(myConfig)
        time.sleep(0.01)
        min_delay=duration*1e6
    else:
        min_delay=10000
    
    ts = get_fpga_time(board=board) + .2e6
    send_virtual_events(board=board, virtual_events=input_events, offset=int(ts), min_delay=int(min_delay))
    output_events = [[], []]
    get_events(board=board, extra_time=1000, output_events=output_events)
    drain_neurons(myConfig,model)
    rates=spike_count(output_events=output_events,show=False)
    pop_rates(rates,test_config)
    [cv_values,synchrony_values]=run_dynamic_anal(output_events,test_config)
    test_config['raster_title']='fI:'+str(in_Freq)
    if any(rates[nvn+1:pcn+nvn+pvn+sstn]>1):
        script_annotated_raster_plot(test_config,output_events,neuron_config,cv_values=cv_values,syn_values=synchrony_values,save_mult=True,annotate=False)
        overtake=frequency_over_time(test_config,output_events)
    else:
        overtake=1
    return rates,cv_values,synchrony_values,overtake


def pop_rates(rates,test_config,show=False):
    nvn=test_config['nvn']
    pcn=test_config['pcn']
    pvn=test_config['pvn']
    sstn=test_config['sstn']

    PC_rates=[0]
    PV_rates=[0]
    SST_rates=[0]
    if len(rates)>1:
        if pcn>0:
            PC_rates=rates[nvn+1:pcn+nvn+1]
        if pvn>0:
            PV_rates=rates[pcn+nvn+1:pcn+pvn+nvn+2]
        if sstn>0:
            SST_rates=rates[pcn+pvn+nvn+2:pcn+pvn+sstn+nvn+2]
    else:
        PC_rates=[0]*pcn
        PV_rates=[0]*pvn
        SST_rates=[0]*sstn

    if show==True:
        print('rates of virtual neurons ')
        print(rates[:nvn])
        print('rates of PC:')
        print(PC_rates)
        print('rates of PV:')
        print(PV_rates)
        print('rates of SST:')
        print(SST_rates)




    return [PC_rates,PV_rates,SST_rates]
 

def set_configs(myConfig,model,neuron_config):
   # PC neuron parameters
   c=0
   #set neuron parameters 
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N",neuron_config['PC_GAIN'][0], neuron_config['PC_GAIN'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", neuron_config['PC_LEAK'][0],  neuron_config['PC_LEAK'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", neuron_config['PC_REF'][0],  neuron_config['PC_REF'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", neuron_config['PC_SPK_THR'][0], neuron_config['PC_SPK_THR'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", neuron_config['PC_DC'][0], neuron_config['PC_DC'][1])
   #set adaptation parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SOAD_PWTAU_N', neuron_config['PC_SOAD_PWTAU_N'][0], neuron_config['PC_SOAD_PWTAU_N'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SOAD_GAIN_P', neuron_config['PC_SOAD_GAIN_P'][0], neuron_config['PC_SOAD_GAIN_P'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SOAD_TAU_P', neuron_config['PC_SOAD_TAU_P'][0],  neuron_config['PC_SOAD_TAU_P'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SOAD_W_N', neuron_config['PC_SOAD_W_N'][0], neuron_config['PC_SOAD_W_N'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SOAD_CASC_P', neuron_config['PC_SOAD_CASC_P'][0],neuron_config['PC_SOAD_CASC_P'][1])
   #set ampa synapse parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', neuron_config['PC_AMPA_TAU'][0],  neuron_config['PC_AMPA_TAU'][1])#Input to PC
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', neuron_config['PC_AMPA_GAIN'][0],  neuron_config['PC_AMPA_GAIN'][1])
   #set nmda synapse parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', neuron_config['PC_NMDA_TAU'][0],  neuron_config['PC_NMDA_TAU'][1])#PC to PC
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', neuron_config['PC_NMDA_GAIN'][0],  neuron_config['PC_NMDA_GAIN'][1])
   #set shunt synapse parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_ITAU_P', neuron_config['PC_SHUNT_TAU'][0],  neuron_config['PC_SHUNT_TAU'][1])# PV to PC
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', neuron_config['PC_SHUNT_GAIN'][0],  neuron_config['PC_SHUNT_GAIN'][1])
   #set gaba synapse parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_ITAU_P', neuron_config['PC_GABA_TAU'][0],  neuron_config['PC_GABA_TAU'][1])# SST to PC
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_IGAIN_P',neuron_config['PC_GABA_GAIN'][0],  neuron_config['PC_GABA_GAIN'][1])
   #set weights
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', neuron_config['PC_W0'][0], neuron_config['PC_W0'][1]) #ampa inpit
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', neuron_config['PC_W1'][0], neuron_config['PC_W1'][1]) #ampa recurrent
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', neuron_config['PC_W2'][0], neuron_config['PC_W2'][1]) #inhibtion PV
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', neuron_config['PC_W3'][0], neuron_config['PC_W3'][1]) #inhibtion SST
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 200)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   # PVs
   c=1
   # Set neuron parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N",neuron_config['PV_GAIN'][0], neuron_config['PV_GAIN'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", neuron_config['PV_LEAK'][0],  neuron_config['PV_LEAK'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", neuron_config['PV_REF'][0],  neuron_config['PV_REF'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", neuron_config['PV_SPK_THR'][0], neuron_config['PV_SPK_THR'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", neuron_config['PV_DC'][0], neuron_config['PV_DC'][1])
   # Set short term depression
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_STDW_N', neuron_config['SYAM_STDW_N'][0],  neuron_config['SYAM_STDW_N'][1]) #short term depression
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAW_STDSTR_N', neuron_config['SYAW_STDSTR_N'][0],  neuron_config['SYAW_STDSTR_N'][1])
   # Set AMPA synapse parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', neuron_config['PV_AMPA_TAU'][0], neuron_config['PV_AMPA_TAU'][1]) # input signal and input SST
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', neuron_config['PV_AMPA_GAIN'][0], neuron_config['PV_AMPA_GAIN'][1])
   # Set gaba synapse parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_ITAU_P', neuron_config['PV_GABA_TAU'][0], neuron_config['PV_GABA_TAU'][1]) #self inhibition and SST inhibition
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_IGAIN_P', neuron_config['PV_GABA_GAIN'][0],  neuron_config['PV_GABA_GAIN'][1])
   # Set shunt synapse parameters  
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_ITAU_P', neuron_config['PV_SHUNT_TAU'][0],  neuron_config['PV_SHUNT_TAU'][1]) #should not be connected
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', neuron_config['PV_SHUNT_GAIN'][0],  neuron_config['PV_SHUNT_GAIN'][1])
   # Set weights
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', neuron_config['PV_W0'][0], neuron_config['PV_W0'][1]) #ampa inpit
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', neuron_config['PV_W1'][0], neuron_config['PV_W1'][1]) #ampa PC exit
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', neuron_config['PV_W2'][0], neuron_config['PV_W2'][1]) #self inhibition
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', neuron_config['PV_W3'][0], neuron_config['PV_W3'][1]) #inhibtion SST
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 200)
   #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   # SST
   c=2
   # Set neuron parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N",neuron_config['SST_GAIN'][0], neuron_config['SST_GAIN'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", neuron_config['SST_LEAK'][0],  neuron_config['SST_LEAK'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", neuron_config['SST_REF'][0],  neuron_config['SST_REF'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", neuron_config['SST_SPK_THR'][0], neuron_config['SST_SPK_THR'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", neuron_config['SST_DC'][0], neuron_config['SST_DC'][1])
   # Set AMPA synapse parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', neuron_config['SST_AMPA_TAU'][0], neuron_config['SST_AMPA_TAU'][1])#PC to SST
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', neuron_config['SST_AMPA_GAIN'][0], neuron_config['SST_AMPA_GAIN'][1])
   # Set gaba synapse parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_ITAU_P', neuron_config['SST_GABA_TAU'][0], neuron_config['SST_GABA_TAU'][1]) #should not be connected
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DEGA_IGAIN_P', neuron_config['SST_GABA_GAIN'][0],  neuron_config['SST_GABA_GAIN'][1])
   # Set shunt synapse parameters
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_ITAU_P', neuron_config['SST_SHUNT_TAU'][0],  neuron_config['SST_SHUNT_TAU'][1])# PV to SST
   set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', neuron_config['SST_SHUNT_GAIN'][0],  neuron_config['SST_SHUNT_GAIN'][1])
   # Set weights
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', neuron_config['SST_W0'][0], neuron_config['SST_W0'][1]) #ampa input, should no occur
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', neuron_config['SST_W1'][0], neuron_config['SST_W1'][1]) #pc input
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', neuron_config['SST_W2'][0], neuron_config['SST_W2'][1]) #pv inhibition
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', neuron_config['SST_W3'][0], neuron_config['SST_W3'][1])
   set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 200)
   model.apply_configuration(myConfig)
   time.sleep(0.2)

def set_all_monitors(number_of_chips,myConfig,model,neuron=10):
    print("Setting monitors")
    for h in range(number_of_chips):
        for c in range(4):
           
            myConfig.chips[h].cores[c].neuron_monitoring_on = True
            myConfig.chips[h].cores[c].monitored_neuron = neuron  # monitor neuron 10 on each core
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    return

def sweep_run(neuron_config,myConfig,model,test_config,input1,board):
    nvn=test_config['nvn']
    pcn=test_config['pcn']
    pvn=test_config['pvn']
    sstn=test_config['sstn']
    sweep_variable=neuron_config['sweep_variable']
    sweep_range=neuron_config['sweep_range_fine']
    in_freq=neuron_config['in_freq']
    in_DC=neuron_config['in_DC']
    sweep_coarse_val=neuron_config['sweep_coarse_val']
    test_config['coarse']=sweep_coarse_val
    duration=neuron_config['duration']

    sweep_fot_output=[]
    sweep_rates_output=[]
    sweep_decay_output=[]
    decay_times=[0,0,0,0]
    for i in sweep_range:
        #update raster title for infividual step
        test_config['raster_title']='Sweep: '+sweep_variable+' '+str([sweep_coarse_val,int(i)])
        #update sweeping parameter
        neuron_config[sweep_variable]=[sweep_coarse_val,int(i)]
        print('Sweeping: '+sweep_variable+' '+str([sweep_coarse_val,int(i)]))
        set_configs(myConfig,model,neuron_config)
        #create input train
        input_events=create_events(input1,nvn,neuron_config,in_freq,duration)
        if neuron_config['input_type']=='DC':
            DC_input(myConfig,model,in_DC)
        #obtain output events
        output_events=run_dynapse(neuron_config,board,input_events)
        rates=spike_count(output_events=output_events,show=False)
        population_rates=pop_rates(rates,test_config)
        sweep_rates_output.append(population_rates)

        #if the rates exist 
        if any(rates[nvn+1:pcn+nvn+pvn+sstn]>1):
            annotated_raster_plot(test_config,output_events,neuron_config,save_mult=True,show=False,annotate=False,annotate_network=False)
            fot_output=frequency_over_time(test_config,output_events)
            frequency_vs_time_plot(fot_output,test_config,show=False)
            sweep_fot_output.append(fot_output)
            #obtain the time to decay
            if neuron_config['decay']==True:
                decay_times=fot_decay(fot_output)
                sweep_decay_output.append(decay_times)
    #save the sweep rates and fot
    sweep_rates_output=np.array(sweep_rates_output,dtype='object')
    np.save(test_config['dir_path']+"/sweep_rates_"+test_config['time_label'],  sweep_rates_output)
    np.save(test_config['dir_path']+"/fot_"+test_config['time_label'],  sweep_fot_output)
    np.save(test_config['dir_path']+"/sweep_range_"+test_config['time_label'],  sweep_range)
    np.save(test_config['dir_path']+"/decay_times"+test_config['time_label'],  sweep_decay_output)
    #plot the sweep rates and fot
    sweep_frequency_vs_parameter_plot(sweep_rates_output,test_config)
    sweep_frequency_vs_time_plot(sweep_fot_output,test_config)

    if neuron_config['decay']==True:
        decay_grap(sweep_range,sweep_decay_output,sweep_variable,test_config)
        np.save(test_config['dir_path']+"/pc_decay_"+test_config['time_label'],  sweep_decay_output)

def config_handshake(neuron_config,time_label,dir_path,config_path,plot_path,date_label,raster_path,tname):
    
    test_config={'duration':neuron_config['duration'],
             'in_DC':neuron_config['in_DC'],
             'in_freq':neuron_config['in_freq'],
             'input_type':neuron_config['input_type'],
             'sweep_variable':neuron_config['sweep_variable'],
             'sweep_coarse_val':neuron_config['sweep_coarse_val'],
             'sweep_range_fine':neuron_config['sweep_range_fine'],
             'nvn':neuron_config['nvn'],
             'pvn':neuron_config['pvn'],
             'pcn':neuron_config['pcn'],
             'sstn':neuron_config['sstn'],
             'time_label':time_label,
             'dir_path':dir_path,
             'config_path':config_path,
             'plot_path':plot_path,
             'date_label':date_label,
             'raster_path':raster_path,
             'test_name':tname
    }
     
    return test_config
    
    
 
    