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


def neuron_configs():
    neuron_config={
        'input_type':'DC',#regular, striated, or sweep
        'DC_Latches':True,
        'overtake_test':False,
        'decay':False,
        'duration':1, #duration of simulation in seconds
        'in_freq':100,#for single neuron tests
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
        #sweep settings
        'sweep':False,#sweep a parameter for single cell tests
        'sweep_variable':'SYAM_STDW_N',#variable to sweep
        'sweep_coarse_val':0,
        'sweep_range_fine':np.linspace(0,250,5,dtype=int),
        #Main Settings
        'PC_Adaptation':False,
        'STD':False,
        #Probabilities
        'Input_PC':.8,
        'Input_PV':.1,
        'Input_SST':1,
        'PC_PC':.2,
        'PC_PV':.2,
        'PC_SST':.05,
        'PV_PC':.2,
        'PV_PV':.2,
        'PV_SST':.2,
        'SST_PC':.6,
        'SST_PV':.2,
        #PC neuron parameters
        'PC_GAIN':[1,46],
        'PC_LEAK':[1,46],#1,100
        'PC_REF':[1,110],
        'PC_SPK_THR':[3,100],
        'PC_DC':[0,0],
        #PC adaptation parameters
        'PC_SOAD_PWTAU_N':[2,150],
        'PC_SOAD_GAIN_P':[4,80],
        'PC_SOAD_TAU_P':[0,200],
        'PC_SOAD_W_N':[3,30],
        'PC_SOAD_CASC_P':[5,250],
        #PC input weight and synapse parameters
        'PC_AMPA_TAU':[1, 60], #5 msinput ampa synapse timesconat 
        'PC_AMPA_GAIN':[3,100], 
        #PC self excitation [NMDA] 
        'PC_NMDA_TAU':[0,20],
        'PC_NMDA_GAIN':[2,125],
        #PC GABA Inhibition from SST [GABA B]
        'PC_GABA_TAU':[1,60],
        'PC_GABA_GAIN':[3,100],
        #PC SHUNT Inhibition from PV [GABA A]
        'PC_SHUNT_TAU':[1,200],
        'PC_SHUNT_GAIN':[3,248],
        #PC Weights
        'PC_W0':[5,105], #input
        'PC_W1':[4,47], #recurrent PC tp PC
        'PC_W2':[1,158], #PV shunt inhitbiton to PC
        'PC_W3':[4,250], #SST shunt inhibition
        #||||||||||||||||||||||||||||||||||||||||||||||||||
        #PV neuron parameters
        'PV_GAIN':[5,250],
        'PV_LEAK':[1,90],
        'PV_REF':[1,200],
        'PV_SPK_THR':[2,160],
        'PV_DC':[0,0],
        #PV input weight and synape parameters
        'PV_AMPA_TAU':[2,60], # 5ms input ampta synapse timecosntan, maybe slower
        'PV_AMPA_GAIN':[4,200], #
        #PV self gaba  inhibiton from both PV and SST GABA B
        'PV_GABA_TAU':[1,100],
        'PV_GABA_GAIN':[4,0],
        #PV Inhibition not set [GABA A]
        'PV_SHUNT_TAU':[1,200],
        'PV_SHUNT_GAIN':[1,100],
        #PV Shtort term depression
        'SYAM_STDW_N':[4,255],
        'SYAW_STDSTR_N':[0,10],
        #PV Weights 
        'PV_W0':[5,200], #input
        'PV_W1':[3,106], # PC input
        'PV_W2':[2,233], # PV gaba inhitbiton
        'PV_W3':[4,150], # SST shunt inhibition
        #||||||||||||||||||||||||||||||||||||||||||||||||||
        #SST neuron parameters
        'SST_GAIN':[1,120],
        'SST_LEAK':[1,44],
        'SST_REF':[1,140],
        'SST_SPK_THR':[3,100],
        'SST_DC':[0,0],
        #SST input weight and synape parameters
        'SST_AMPA_TAU':[2,60],
        'SST_AMPA_GAIN':[5,200], 
        #SST self gaba  inhibiton from PV GABA B
        'SST_GABA_TAU':[1,100],
        'SST_GABA_GAIN':[2,120],
        #SST Inhibition from SST GABA A
        'SST_SHUNT_TAU':[1,200],
        'SST_SHUNT_GAIN':[3,120],
        #SST Weights
        'SST_W0':[3,100], #input
        'SST_W1':[0,0], #PC input
        'SST_W2':[0,187], # pv gaba inhitbiton
        'SST_W3':[0,0],# nothing
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


