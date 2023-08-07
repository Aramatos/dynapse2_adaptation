from cgi import test
from pickle import TRUE
import numpy as np
import itertools
#import pandas as pd
import os

import re
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.ticker as ticker
import matplotlib.colors as colors

def weighted_moving_average(signal, times, window_size):
    """
    Compute a weighted moving average of a signal.

    Args:
    signal: numpy array of shape (n,)
    times: numpy array of shape (n,) representing the time points corresponding to the signal
    window_size: integer representing the size of the moving window

    Returns:
    numpy array of shape (n,) representing the weighted moving average of the signal
    """
    # Create a weighting kernel
    weights = np.exp(-np.linspace(-1, 1, window_size)**2)

    # Normalize the kernel to sum to 1
    weights /= weights.sum()

    # Reflect the signal at the edges
    reflected_signal = np.pad(signal, (window_size//2, window_size//2), mode='reflect')

    # Convolve the signal with the weighting kernel
    output = np.convolve(reflected_signal, weights, mode='same')


    # Trim the output to the original signal size
    output = output[(window_size//2):-(window_size//2)]

    return output

def CV_Analysis(nvn,pvn,pcn,sstn,output,show=True):
    output=np.asanyarray(output)
    if len(output[1])>0:
        times=output[1]-output[1][0]
        spike_id=output[0]
        pc_cv_list=[]
        sst_cv_list=[]
        pv_cv_list=[]
        if pcn>0:
            for i in range(nvn+1,pcn+1+nvn):  
                occurences=times[spike_id==i]
                if len(occurences)>1:
                    isi=np.diff(occurences)
                    cv=np.var(isi)/(np.average(isi)**2)
                    pc_cv_list.append(cv)
                else:
                    pc_cv_list.append(0)  

            pc_cv_list=np.asanyarray(pc_cv_list)
            PC_average=np.mean(pc_cv_list)
        else:
            PC_average=0
        if pvn>0:
            for i in range(nvn+pcn+2,pcn+2+pvn+nvn):  
                occurences=times[spike_id==i]
                if len(occurences)>1:
                    isi=np.diff(occurences)
                    cv=np.var(isi)/(np.average(isi)**2)
                    pv_cv_list.append(cv)
                else:
                    pv_cv_list.append(0)  

            pv_cv_list=np.asanyarray(pv_cv_list)
            PV_average=np.mean(pv_cv_list)
        else:
            PV_average=0

        if sstn > 0:
            for i in range(nvn+pcn+pvn+3,pcn+3+pvn+nvn+sstn):  
                occurences=times[spike_id==i]
                if len(occurences)>1:
                    isi=np.diff(occurences)
                    cv=np.var(isi)/(np.average(isi)**2)
                    sst_cv_list.append(cv)
                else:
                    sst_cv_list.append(0)    

            sst_cv_list=np.asanyarray(sst_cv_list)
            SST_average=np.mean(sst_cv_list)
        else:
            SST_average=0
        if show==True:
            
            print('PC_CV_average: '+str(np.round(PC_average,2))+' PV_CV_average: '+str(np.round(PV_average,2))+' SST_CV_average: '+str(np.round(SST_average,2)) )    
    else:
        PC_average=0
        PV_average=0
        SST_average=0
    return [PC_average,PV_average,SST_average]  

def frequency_arrays(output_events, test_config):
    output_events=np.asanyarray(output_events)
    spike_times_all=output_events[1]-output_events[1][0]
    spike_id=output_events[0]
    pc_ff_list=[]
    pc_time_list=[]
    pvn_ff_list=[]
    pvn_time_list=[]
    sstn_ff_list=[]
    sstn_time_list=[]
    nvn=test_config['nvn']
    pcn=test_config['pcn']
    pvn=test_config['pvn']
    sstn=test_config['sstn']
    if pcn>0:
        for i in range(nvn+1,pcn+1+nvn):  
                    occurences=spike_times_all[spike_id==i]
                    ocurrence_times=spike_times_all[spike_id==i]
                    if len(occurences)>0:
                        isi=np.diff(occurences)
                        if(isi==0).any():
                            print("removed")
                        else:
                            #numpy append to isi to ff list
                            pc_ff_list.append(1/isi)
                            pc_time_list.append(ocurrence_times)
    if pvn>0:
        for i in range(nvn+pcn+1,nvn+pcn+pvn+1):
                    occurences=spike_times_all[spike_id==i]
                    ocurrence_times=spike_times_all[spike_id==i]
                    if len(occurences)>0:
                        isi=np.diff(occurences)
                        if(isi==0).any():
                            print("removed")
                        else:
                            #numpy append to isi to ff list
                            pvn_ff_list.append(1/isi)
                            pvn_time_list.append(ocurrence_times)
    if sstn>0:
        for i in range(nvn+pcn+pvn+1,nvn+pcn+pvn+sstn+1):
                    occurences=spike_times_all[spike_id==i]
                    ocurrence_times=spike_times_all[spike_id==i]
                    if len(occurences)>0:
                        isi=np.diff(occurences)
                        if(isi==0).any():
                            print("removed")
                        else:
                            #numpy append to isi to ff list
                            sstn_ff_list.append(1/isi)
                            sstn_time_list.append(ocurrence_times)
    return pc_ff_list, pc_time_list, pvn_ff_list, pvn_time_list, sstn_ff_list, sstn_time_list


def split_spike_indices_by_window(times, indices, window_size:int, total_duration:int):
	"""
	Split a list of spike indices (with accompanying spike times) into bins 
	based on equal length time windows.
	"""
	return np.split(indices, np.sum(np.atleast_2d(times) < np.atleast_2d(np.arange(window_size,total_duration,window_size)).T, axis=1))

def Synchrony_Analsis(nvn,pcn,pvn,sstn,output,analysis_time,show=True):
    output=np.asanyarray(output)
    times=output[1]-output[1][0]
    spike_id=np.asanyarray(output[0],dtype=np.uint8)

    pc_id=spike_id[(spike_id>nvn+1)&(spike_id<nvn+1+pcn)]
    pc_times=times[(spike_id>nvn+1)&(spike_id<nvn+1+pcn)]

    pv_id=spike_id[(spike_id>nvn+1+pcn)&(spike_id<nvn+2+pcn+pvn)]
    pv_times=times[(spike_id>nvn+1+pcn)&(spike_id<nvn+2+pcn+pvn)]


    if sstn>0:
        sst_id=spike_id[(spike_id>nvn+2+pcn+pvn)&(spike_id<nvn+3+pcn+pvn+sstn)]
        sst_times=times[(spike_id>nvn+2+pcn+pvn)&(spike_id<nvn+3+pcn+pvn+sstn)]
        SST_synchrony = spike_train_synchrony_correlation(sst_times,sst_id,analysis_time)
        PC_synchrony = spike_train_synchrony_correlation(pc_times,pc_id,analysis_time)
        PV_synchrony = spike_train_synchrony_correlation(pv_times,pv_id,analysis_time)

    else:
        SST_synchrony=0
        PC_synchrony = spike_train_synchrony_correlation(pc_times,pc_id,analysis_time)
        PV_synchrony = spike_train_synchrony_correlation(pv_times,pv_id,analysis_time)

    
    if show == True:
        print('PC_synchrony: '+str(round(PC_synchrony,2))+' PV_CV_average: '+str(round(PV_synchrony,2))+' SST_CV_average: '+str(round(SST_synchrony,2)) )    

    return [PC_synchrony,PV_synchrony,SST_synchrony]
    

def spike_train_synchrony_correlation(spike_times, spike_indices, total_duration:int):
	"""
	Calculate the average correlation between the windowed spike counts for all pairs
	of neurons in the network.
	Returns NaN if no neurons fired. Returns 0 if only one neuron fired.

	"""
	if len(spike_indices) == 0:
		return np.nan
	elif len(spike_indices) == 1:
		return 0
	
	neuron_indices = np.unique(spike_indices)
	num_neurons = len(neuron_indices)
	windowed_spikes = split_spike_indices_by_window(spike_times, spike_indices, .005, total_duration)
	windowed_spike_counts = np.zeros((1+np.max(neuron_indices), len(windowed_spikes)))

	for i, window in enumerate(windowed_spikes):
		spike_indices, spike_counts = np.unique(window, return_counts=True)
		windowed_spike_counts[spike_indices, i] = spike_counts
	
	# only calculate the correlations for spike trains where we have spikes
	spiking_neuron_spike_trains = windowed_spike_counts[neuron_indices,:]
	if spiking_neuron_spike_trains.shape[0] > 1:
		correlation_coefficients = np.tril(np.corrcoef(spiking_neuron_spike_trains), k=-1)
		num_coefficients = num_neurons*(num_neurons-1) / 2
		# on the off chance that a spike train is perfectly regular, correlating with it will
		# result in a NaN value - remove these
		return np.nansum(correlation_coefficients) / (num_coefficients - np.count_nonzero(np.isnan(correlation_coefficients)))
	else:
		# we only have one neuron spike train - no synchronisation
		return 0.0




def run_dynamic_anal(output_events,test_config):    
    nvn=test_config['nvn']
    pcn=test_config['pcn']
    pvn=test_config['pvn']
    sstn=test_config['sstn']
    if len(output_events[1])<1:
        cv_values=[0,0,0]
        synchrony_values=[1,1,1]
        return cv_values,synchrony_values
    else:
        cv_values=CV_Analysis(nvn,pvn,pcn,sstn,output_events,show=True)
        synchrony_values=Synchrony_Analsis(nvn,pcn,pvn,sstn,output_events,1,show=True)
    return cv_values,synchrony_values

def analysis_error(test_config,cv_values,synchrony_values):
    pcn=test_config['pcn']
    pvn=test_config['pvn']
    sstn=test_config['sstn']
    if (pvn>0 and pcn>0 and sstn>0):
        cv_error=round((abs(cv_values[0]-1)+abs(cv_values[1]-1)+abs(cv_values[2]-1))/3,3)
        synchrony_error=round(((synchrony_values[0])+abs(synchrony_values[1])+abs(synchrony_values[2]))/3,3)
        error=(cv_error+synchrony_error)/2
    elif (pvn>0 and pcn>0):
        cv_error=round((abs(cv_values[0]-1)+abs(cv_values[1]-1))/2,3)
        synchrony_error=round(((synchrony_values[0])+abs(synchrony_values[1]))/2,3)
        error=(cv_error+synchrony_error)/2
    else: 
        print('Variability and Synchony Analsis failed due to incorrect neuron initialization')
    return error,cv_error,synchrony_error


def frequency_over_time(test_config,output_events):
    '''
    This function calculates the firing frequency of each neuron ggroup over time as ana everage over bin times

    '''
    nvn=test_config['nvn']
    pcn=test_config['pcn']
    pvn=test_config['pvn']
    sstn=test_config['sstn']
    duration=test_config['duration']
    output_events=np.asanyarray(output_events)
    times=output_events[1]-output_events[1][0]
    spike_id=output_events[0]
    step=.040
    
    if pcn>0:
        pc_id=spike_id[(spike_id>nvn+1)&(spike_id<nvn+1+pcn)]
        pc_times=times[(spike_id>nvn+1)&(spike_id<nvn+1+pcn)]

        windows_pc=split_spike_indices_by_window(pc_times, pc_id, step, duration)
        spikes_pc=[len(windows_pc[i]) for i in range(len(windows_pc))]
        ff_windows_pc=(np.asarray(spikes_pc,dtype=float)/pcn/step)[:-1]
    else:
        ff_windows_pc=np.zeros(len(np.arange(0,duration,step)))[:-1]

    if pvn>0:
        pv_id=spike_id[(spike_id>nvn+1+pcn)&(spike_id<nvn+2+pcn+pvn)]
        pv_times=times[(spike_id>nvn+1+pcn)&(spike_id<nvn+2+pcn+pvn)]

        windows_pv=split_spike_indices_by_window(pv_times, pv_id, step, duration)
        spikes_pv=[len(windows_pv[i]) for i in range(len(windows_pv))]
        ff_windows_pv=(np.asarray(spikes_pv,dtype=float)/pvn/step)[:-1]
    else:
        ff_windows_pv=np.zeros(len(np.arange(0,duration,step)))[:-1]

    if sstn>0:
        sst_id=spike_id[(spike_id>nvn+2+pcn+pvn)&(spike_id<nvn+3+pcn+pvn+sstn)]
        sst_times=times[(spike_id>nvn+2+pcn+pvn)&(spike_id<nvn+3+pcn+pvn+sstn)]
        
        windows_sst=split_spike_indices_by_window(sst_times, sst_id, step, duration)
        spikes_sst=[len(windows_sst[i]) for i in range(len(windows_sst))]
        ff_windows_sst=(np.asarray(spikes_sst,dtype=float)/sstn/step)[:-1]

    else:
        ff_windows_sst=np.zeros(len(np.arange(0,duration,step)))[:-1]

    time_axis=np.arange(0,duration,step)[:-1]


    return [time_axis,ff_windows_pc,ff_windows_pv,ff_windows_sst]

    #//////////////////////////////////////////



def SST_Overtake_time(fot_output):
    time_overtake=fot_output[0][fot_output[2]>fot_output[1]][0]
    return time_overtake

def fot_decay(fot_output):
    '''
    This function calculates the decay time of the frequency over time curve for each neuron group
    utilizing the 38% of the final frequency as the cutoff point. It does start at the maximum frequency
    and works backwards to find the first time point that is below the cutoff point.
    '''
    time_axis=fot_output[0]
    decay_times=[0,0,0]
    for i in range(1,4):
        freq_list=fot_output[i]
        if any(freq_list)>0:
            m_i=np.argmax(freq_list)
            f_ss=freq_list[-1]
            freq_list_2=freq_list[m_i:]-f_ss
            time_axis_2=time_axis[m_i:]
            f_i=np.max(freq_list)*.38
            decay_times[i-1]=time_axis_2[freq_list_2<f_i][0]
        else:
            decay_times[i-1]=0
    print(decay_times)
    return decay_times


# Assuming spike_times_pcn is a list of spike times for the 'pcn' spike train
# Example: spike_times_pcn = [np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.4, 0.6])]

def psth_calc(spike_times, bin_size=0.011,duration=1):

    """Calculate the peristimulus time histogram (PSTH) for a given spike train.
    Parameters
    ----------
    spike_times : list of numpy arrays
        Spike times for each trial.
    bin_size : float
        Bin size in seconds.
    Returns
    -------
    psth : numpy array
        PSTH values.
    bins : numpy array
        Bin edges.
    """
    # Calculate the number of trials
    n_trials = len(spike_times)
    # Calculate the bins
    bins = np.arange(0, duration, bin_size)
    # Calculate the PSTH
    psth = np.zeros(len(bins) - 1)
    for i in range(n_trials):
        psth += np.histogram(spike_times[i], bins=bins)[0]
    psth = psth / (n_trials * bin_size)
    return psth, bins
    

def spike_time_arrays(output_events,nvn=0,pvn=0,pcn=0,sstn=0):
    output_events=np.asanyarray(output_events)
    spike_times_all=output_events[1]-output_events[1][0]
    neuron_indexes=output_events[0]
    unique_neurons = np.unique(neuron_indexes)
    spike_times = [spike_times_all[neuron_indexes == i] for i in unique_neurons]
    spike_times = np.asanyarray(spike_times,dtype=object)
    spike_times_nvn=spike_times[0:nvn]
    spike_times_pvn=spike_times[nvn:nvn+pvn]
    spike_times_pcn=spike_times[nvn+pvn:nvn+pvn+pcn]
    spike_times_sstn=spike_times[nvn+pvn+pcn:nvn+pvn+pcn+sstn]
    return spike_times_nvn,spike_times_pvn,spike_times_pcn,spike_times_sstn

def spike_time_array(output_events, nvn, pvn, pcn, sstn, neuron_type):
    # Convert input to numpy array
    output_events = np.asarray(output_events)

    # Calculate spike times relative to the first spike
    spike_times_all = output_events[1] - output_events[1][0]

    # Get neuron indexes from the input
    neuron_indexes = output_events[0]

    # Get unique neurons from the input
    unique_neurons = np.unique(neuron_indexes)

    # Get spike times for each neuron
    spike_times = [spike_times_all[neuron_indexes == i] for i in unique_neurons]

    # Convert spike times to a numpy array with object data type
    spike_times = np.asarray(spike_times, dtype=object)

    # Set spike_times_type to the spike times for the specified neuron type
    if neuron_type == "nvn":
        spike_times_type = spike_times[0:nvn]
    elif neuron_type == "pvn":
        spike_times_type = spike_times[nvn:nvn+pvn]
    elif neuron_type == "pcn":
        spike_times_type = spike_times[nvn+pvn:nvn+pvn+pcn]
    elif neuron_type == "sstn":
        spike_times_type = spike_times[nvn+pvn+pcn:nvn+pvn+pcn+sstn]
    else:
        # If an invalid neuron type is specified, return an error message
        return "Invalid neuron type"

    # Return spike times for the specified neuron type
    return spike_times_type


