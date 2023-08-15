from cgi import test
import numpy as np

def test_selection():
    test_select={
        'FF_Network':False,
        'FF_PC_PV':False,
        'FF_Single_Neurons':True,
        'PC_PV_SST_Network':False,
        'PC_PV_Network':False,
        'PC_Neuron':False,
        'PV_Neuron':True,
        'SST_Neuron':False,
        'DE_PV_PC': False,
    }
    return test_select


def config():
    neuron_config={
        'input_type':'Regular',#regular, striated, or sweep
        #General Latch Settings
        'PC_Adaptation':False,
        'STD':False,
        'DC_Latches':False,
        'overtake_test':False,
        'decay':False,
        'duration':1, #duration of simulation in seconds
        'in_freq':200,#for single neuron tests
        'in_DC':0,#for single neuron tests
        'plot_iter':False,#plot random search graphs
        'n_iter':20,#for Random searches
        #Striated input settings
        'Striated':False,#Striated
        'duration1':300000,#duration of first input in us
        'duration2':300000,#duration of second input in us
        'rest_time':200000,#striated input rest time
        #oscilloscope test settings
        'core_to_measure':0,
        'synapse_to_measure':'ampa',
        'neuron_to_measure':10,
        'test_type':'neuron',
        'DC_pulse':[3,250],
        #sweep parameter settings
        'sweep':False,#sweep a parameter for single cell tests
        'sweep_variable':'SYAM_STDW_N',#variable to sweep
        'sweep_coarse_val':0,
        'sweep_range_fine':np.linspace(0,250,5,dtype=int),
        #F-I settings
        'DC_Coarse':2,
        'DC_FI_Range':np.linspace(0,250,10,dtype=int),
        'Freq_FI_Range:':np.linspace(0,250,10,dtype=int),
        #population size
        'nvn':10,
        'pcn':80,
        'pvn':10,        
        'sstn':10,
        #Probabilities
        'Input_PC':.3,
        'Input_PV':.11,
        'Input_SST':.1,
        'PC_PC':.2,
        'PC_PV':.15,
        'PC_SST':.3,
        'PV_PC':.2,
        'PV_PV':.2,
        'PV_SST':.3,
        'SST_PC':.3,
        'SST_PV':.2,
        #PC neuron parameters
        'PC_LEAK':[1,46],
        'PC_GAIN':[0,30],
        'PC_REF':[1,70],
        'PC_SPK_THR':[4,100],
        'PC_DC':[0,0],
        #PC adaptation parameters
        'PC_SOAD_PWTAU_N':[2,150],
        'PC_SOAD_GAIN_P':[4,80],
        'PC_SOAD_TAU_P':[0,200],
        'PC_SOAD_W_N':[3,130],
        'PC_SOAD_CASC_P':[5,250],
        #PC input weight and synapse parameters
        'PC_AMPA_TAU':[1, 60], #5 msinput ampa synapse timesconat 
        'PC_AMPA_GAIN':[4,100], 
        #PC self excitation [NMDA] 
        'PC_NMDA_TAU':[0,20],
        'PC_NMDA_GAIN':[3,100],
        #PC GABA Inhibition from SST [GABA B]
        'PC_GABA_TAU':[0,15],
        'PC_GABA_GAIN':[3,100],
        #PC SHUNT Inhibition from PV [GABA A]
        'PC_SHUNT_TAU':[1,30],
        'PC_SHUNT_GAIN':[1,90],
        #PC Weights
        'PC_W0':[4,100], #input
        'PC_W1':[1,100], #recurrent PC tp PC
        'PC_W2':[2,100], #PV shunt inhitbiton to PC
        'PC_W3':[2,100], #SST shunt inhibition
        #||||||||||||||||||||||||||||||||||||||||||||||||||
        #PV neuron parameters
        'PV_LEAK':[1,90],
        'PV_GAIN':[1,70],
        'PV_REF':[1,200],
        'PV_SPK_THR':[5,250],
        'PV_DC':[0,0],
        #PV input weight and synape parameters
        'PV_AMPA_TAU':[1,70], # 5ms input ampta synapse timecosntan, maybe slower
        'PV_AMPA_GAIN':[5,150], #
        #PV self gaba  inhibiton from both PV and SST GABA B
        'PV_GABA_TAU':[0,15],
        'PV_GABA_GAIN':[3,100],
        #PV Inhibition not set [GABA A]
        'PV_SHUNT_TAU':[1,30],
        'PV_SHUNT_GAIN':[1,90],
        #PV Shtort term depression
        'SYAM_STDW_N':[4,255],
        'SYAW_STDSTR_N':[0,10],
        #PV Weights 
        'PV_W0':[3,100], #input
        'PV_W1':[1,100], # PC input
        'PV_W2':[4,250], # PV gaba inhitbiton
        'PV_W3':[2,100], # SST shunt inhibition
        #||||||||||||||||||||||||||||||||||||||||||||||||||
        #SST neuron parameters
        'SST_LEAK':[1,44],
        'SST_GAIN':[1,80],
        'SST_REF':[1,90],
        'SST_SPK_THR':[5,250],
        'SST_DC':[0,0],
        #SST input weight and synape parameters
        'SST_AMPA_TAU':[1,60],
        'SST_AMPA_GAIN':[5,200], 
        #SST self gaba  inhibiton from PV GABA B
        'SST_GABA_TAU':[0,15],
        'SST_GABA_GAIN':[3,100],
        #SST Inhibition from SST GABA A
        'SST_SHUNT_TAU':[1,30],
        'SST_SHUNT_GAIN':[1,90],
        #SST Weights
        'SST_W0':[4,100], #input
        'SST_W1':[4,200], #PC input
        'SST_W2':[2,100], # pv gaba inhitbiton
        'SST_W3':[0,0],# nothing is connected here
        }
    return neuron_config

'''
def test_config():
    test_config={
        'duration':duration,
        'in_DC':in_DC,
        'tname':tname,
        'in_freq':in_freq,
        'nvn':nvn,
        'pvn':pvn,
        'pcn':pcn,
        'sstn':sstn,
        'time':time_label,
        'config_path':config_path,
        'plot_path':plot_path,
        'date_label':date_label,
        'raster_path':raster_path
        }
'''


