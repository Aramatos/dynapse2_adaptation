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
        'PV_Neuron':False,
        'SST_Neuron':False,
        'DE_PV_PC': False,
    }
    return test_select

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


def neuron_configs():
    neuron_config={
        'input_type':'Striated',
        'DC_Latches':False,
        'overtake_test':False,
        'decay':False,
        'duration':1, #duration of simulation in seconds
        'in_freq':100,#for single neuron tests
        'in_DC':100,#for single neuron tests
        'n_iter':20,#for Random searches
        'rest_time':0,#striated input rest time
        'plot_iter':False,#plot random search graphs
        #sweep settings
        'sweep':False,#sweep a parameter for single cell tests
        'sweep_variable':'SYAW_STDSTR_N',
        'sweep_coarse_val':0,
        'sweep_range_fine':np.linspace(0,50,10),
        #Main Settings
        'PC_Adaptation':False,
        'STD':True,
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
        'PC_GAIN':[4,250],
        'PC_LEAK':[0,50],
        'PC_REF':[1,180],
        'PC_SPK_THR':[2,220],
        'PC_DC':[0,10],
        #PC adaptation parameters
        'PC_SOAD_PWTAU_N':[4,100],
        'PC_SOAD_GAIN_P':[4,50],
        'PC_SOAD_TAU_P':[1,50],
        'PC_SOAD_W_N':[4,50],
        'PC_SOAD_CASC_P':[5,250],
        #PC input weight and synapse parameters
        'PC_AMPA_TAU':[1,130], #5 ms input ampa synapse timesconat 
        'PC_AMPA_GAIN':[3,100],
        #PC self excitation NMDA
        'PC_NMDA_TAU':[1,60],
        'PC_NMDA_GAIN':[3,125],
        #PC GABA Inhibition from SST
        'PC_GABA_TAU':[1,100],
        'PC_GABA_GAIN':[3,100],
        #PC SHUNT Inhibition from PV
        'PC_SHUNT_TAU':[1,140],
        'PC_SHUNT_GAIN':[3,248],
        #PC Weights
        'PC_W0':[5,120], #input
        'PC_W1':[3,47], #recurrent PC tp PC
        'PC_W2':[4,238], #PV shunt inhitbiton to PC
        'PC_W3':[4,250], #SST shunt inhibition
        #||||||||||||||||||||||||||||||||||||||||||||||||||
        #PV neuron parameters
        'PV_GAIN':[3,180],
        'PV_LEAK':[2,45],
        'PV_REF':[2,35],
        'PV_SPK_THR':[3,160],
        'PV_DC':[0,10],
        #PV input weight and synape parameters
        'PV_AMPA_TAU':[0,100], # $5ms input ampta synapse timecosntan, maybe slower
        'PV_AMPA_GAIN':[1,140], #223
        #PV self gaba  inhibiton from SST
        'PV_GABA_TAU':[1,100],
        'PV_GABA_GAIN':[4,0],
        #PV Inhibition not set
        'PV_SHUNT_TAU':[1,200],
        'PV_SHUNT_GAIN':[1,100],
        #PV Shtort term depression
        'SYAM_STDW_N':[5,255],
        'SYAW_STDSTR_N':[0,6],
        #PV Weights
        'PV_W0':[4,205], #input
        'PV_W1':[3,106], # PC input
        'PV_W2':[3,233], # PV gaba inhitbiton
        'PV_W3':[4,150], # SST shunt inhibition
        #||||||||||||||||||||||||||||||||||||||||||||||||||
        #SST neuron parameters
        'SST_GAIN':[4,180],
        'SST_LEAK':[0,40],
        'SST_REF':[2,60],
        'SST_SPK_THR':[3,160],
        'SST_DC':[0,0],
        #SST input weight and synape parameters
        'SST_AMPA_TAU':[0,80],
        'SST_AMPA_GAIN':[2,100], 
        #SST self gaba  inhibiton from PV
        'SST_GABA_TAU':[1,100],
        'SST_GABA_GAIN':[2,120],
        #SST Inhibition from SST
        'SST_SHUNT_TAU':[1,100],
        'SST_SHUNT_GAIN':[3,120],
        #SST Weights
        'SST_W0':[0,20], #input
        'SST_W1':[0,0], #PC input
        'SST_W2':[0,187], # pv gaba inhitbiton
        'SST_W3':[0,0],# nothing
        }
    return neuron_config


