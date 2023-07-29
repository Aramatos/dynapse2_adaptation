from imp import init_frozen
from pickle import TRUE
from pickletools import uint8
import time
import sys
import os
import dwf
import numpy as np
import threading


from venv import create

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_network import Network
from lib.dynapse2_spikegen import send_events,get_fpga_time, send_virtual_events, poisson_gen, isi_gen,regular_gen
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *
from adaptation_lib.spike_stats import *
from adaptation_lib.dynapse_setup import *
from adaptation_lib.graphing import *
from configs.neuron_configs import neuron_configs
import numpy as np
import matplotlib as mp
import datetime
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt


board_names=["dev_board"]

def change_synapse_leak(myConfig,test_type,model,core_to_measure,coarse,fine):
    #input_events=regular_gen(input_neuron,1,10,1)
    if test_type=='AMPA':
        #set ampa synapse parameters
        set_parameter(myConfig.chips[0].cores[core_to_measure].parameters,'DEAM_ETAU_P',coarse,fine)
    elif test_type=='NMDA':    
        #set nmda synapse parameters
        set_parameter(myConfig.chips[0].cores[core_to_measure].parameters, 'DENM_ETAU_P', coarse,fine)#PC to PC
    elif test_type=='GABA_B':
        #set GABA B slow inhibitory substractive synapse parameters
        set_parameter(myConfig.chips[0].cores[core_to_measure].parameters, 'DEGA_ITAU_P',coarse,fine)# SST to PC
    elif test_type=='GABA_A':
        #set GABA A fast inhibitory shunt synapse parameters 
        set_parameter(myConfig.chips[0].cores[core_to_measure].parameters, 'DESC_ITAU_P',coarse,fine)# PV to PC
    model.apply_configuration(myConfig)
    time.sleep(0.1)


def change_synapse_gain(myConfig,test_type,model,core_to_measure,coarse,fine):
    #input_events=regular_gen(input_neuron,1,10,1)
    if test_type=='AMPA':
        #set ampa synapse parameters
        set_parameter(myConfig.chips[0].cores[core_to_measure].parameters,'DEAM_EGAIN_P',coarse,fine)
    elif test_type=='NMDA':    
        #set nmda synapse parameters
        set_parameter(myConfig.chips[0].cores[core_to_measure].parameters, 'DENM_EGAIN_P', coarse,fine)#PC to PC
    elif test_type=='GABA_B':
        #set GABA B slow inhibitory substractive synapse parameters
        set_parameter(myConfig.chips[0].cores[core_to_measure].parameters, 'DEGA_IGAIN_P',coarse,fine)# SST to PC
    elif test_type=='GABA_A':
        #set GABA A fast inhibitory shunt synapse parameters 
        set_parameter(myConfig.chips[0].cores[core_to_measure].parameters, 'DESC_IGAIN_P',coarse,fine)# PV to PC
    model.apply_configuration(myConfig)
    time.sleep(0.1)


def epsp_spike(board,input_events):
    min_delay=10000
    print("\ngetting fpga time\n")
    ts = get_fpga_time(board=board) + 100000
    send_virtual_events(board=board, virtual_events=input_events, offset=int(ts), min_delay=int(min_delay))
    time.sleep(1)
    return

def pulse(board,number_of_chips,neuron_config):
    neurons = range(256)
    board_names=["dev_board"]
    #Initialization
    model = board.get_model()
    model.reset(ResetType.PowerCycle, (1 << number_of_chips) - 1)
    time.sleep(.2)
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    # set current neuron parameters
    print("Setting parameters")
    set_configs(myConfig,model,neuron_config)
    # set neurons spiking threadholds to maximum
    for c in range(4):
        set_parameter(myConfig.chips[0].cores[c].parameters,'SOIF_GAIN_N',0,70)
        set_parameter(myConfig.chips[0].cores[c].parameters,'SOIF_REFR_N',3,110)
        set_parameter(myConfig.chips[0].cores[c].parameters,'SOIF_SPKTHR_P',5,250)
        set_parameter(myConfig.chips[0].cores[c].parameters,'SYPD_EXT_N', 3, 200)
    model.apply_configuration(myConfig)
    # set neurons to monitor
    myConfig.chips[0].cores[neuron_config['core_to_measure']].neuron_monitoring_on = True
    myConfig.chips[0].cores[neuron_config['core_to_measure']].monitored_neuron = neuron_config['neuron_to_measure']  # monitor neuron 10 on each core
    model.apply_configuration(myConfig)
    time.sleep(0.1)
    # set neuron latches to get DC input
    set_dc_latches(config=myConfig, neurons=neurons, cores=range(4), chips=range(number_of_chips))
    print("\nAll configurations done!\n")
    set_DC_parameter(myConfig,model, neuron_config['DC_pulse'][0],neuron_config['DC_pulse'][1])
    time.sleep(.6)
    set_DC_parameter(myConfig,model, 0, 0)
    time.sleep(3)
    return

def normalize_input(input_array, max_value=1.0):
    """
    Function to normalize input data within a specified range.
    
    Parameters:
    input_array (np.ndarray): The input data to be normalized.
    max_value (float): The maximum value for normalization. Default is 1.0.

    Returns:
    np.ndarray: The normalized input data.
    """
    data_min = np.min(input_array)
    data_max = np.max(input_array)

    # Adjust the normalization according to the max_value parameter
    adjusted_data_max = data_min + ((data_max - data_min) * max_value)

    normalized_input = (input_array - data_min) / (adjusted_data_max - data_min)

    return normalized_input

def custom_relu(x, start_rise, slope, saturation):
    """
    Custom piece-wise linear function with a flat portion, a rising slope, and a saturation point.

    Parameters:
        x (float or numpy array): Input value(s).
        start_rise (float): Starting point of the rising slope.
        slope (float): Slope of the rising portion.
        saturation (float): Saturation point where the function plateaus.

    Returns:
        float or numpy array: Output value(s) after applying the custom piece-wise linear function.
    """
    linear_part = lambda x: slope * (x - start_rise)
    saturated_part = lambda x: np.full_like(x, saturation)
    
    return np.piecewise(x, 
                        [x < start_rise, (x >= start_rise) & (x < saturation/slope + start_rise), x >= saturation/slope + start_rise], 
                        [0, linear_part, saturated_part])


def spike_osc_measurement(board,input_events,duration=6):
    # Define constants
    CHANNEL = 0  # Use first channel
    FREQUENCY = 1200.0  # Sampling frequency
    DURATION = duration  # Duration of recording in seconds
    BUFFER_SIZE = int(FREQUENCY * DURATION)  # Number of samples
    try:
        # Create a DWF instance
        analog_instrument = dwf.DwfAnalogIn()

        # Open the first available device
        analog_instrument.channelEnableSet(CHANNEL, True)
        analog_instrument.channelRangeSet(CHANNEL, 5.0)  # Set voltage range to -5V to +5V

        # Set the sample rate and buffer size
        analog_instrument.frequencySet(FREQUENCY)
        analog_instrument.recordLengthSet(DURATION)

        # Wait for the device to settle
        time.sleep(.5)

        # Start the acquisition
        analog_instrument.configure(False, True)

        # Start another function in a separate thread

        threading.Thread(target=epsp_spike(board,input_events)).start()
        # Wait until the acquisition is done
        while True:
            if analog_instrument.status(True) == dwf.DwfStateDone:
                break

        # Get the acquired samples
        samples = analog_instrument.statusData(CHANNEL, BUFFER_SIZE)
        samples=np.array(samples)
        # Generate corresponding time values
        time_values = np.linspace(0, DURATION, num=BUFFER_SIZE, endpoint=False)


    finally:
        # Always close the device, even if an error occurred
        analog_instrument.close()
    return samples,time_values

def pulse_osc_measurement(board,number_of_chips,neuron_config,duration=6,frequency=1200):
  # Define constants
    CHANNEL = 0  # Use first channel
    FREQUENCY = frequency  # Sampling frequency
    DURATION = duration  # Duration of recording in seconds
    BUFFER_SIZE = int(FREQUENCY * DURATION)  # Number of samples
    try:
        # Create a DWF instance
        analog_instrument = dwf.DwfAnalogIn()

        # Open the first available device
        analog_instrument.channelEnableSet(CHANNEL, True)
        analog_instrument.channelRangeSet(CHANNEL, 5.0)  # Set voltage range to -5V to +5V

        # Set the sample rate and buffer size
        analog_instrument.frequencySet(FREQUENCY)
        analog_instrument.recordLengthSet(DURATION)

        # Wait for the device to settle
        time.sleep(.5)

        # Start the acquisition
        analog_instrument.configure(False, True)

        # Start another function in a separate thread

        threading.Thread(target=pulse(board,number_of_chips,neuron_config)).start()
        # Wait until the acquisition is done
        while True:
            if analog_instrument.status(True) == dwf.DwfStateDone:
                break

        # Get the acquired samples
        samples = analog_instrument.statusData(CHANNEL, BUFFER_SIZE)
        samples=np.array(samples)
        # Generate corresponding time values
        time_values = np.linspace(0, DURATION, num=BUFFER_SIZE, endpoint=False)


    finally:
        # Always close the device, even if an error occurred
        analog_instrument.close()
    return samples,time_values

from scipy.signal import savgol_filter

def time_constant_extraction(data, output='tc', beta=1/26, cut_off=15, window_length=51, polyorder=3):
    raw_voltage=data[0]*1000
    raw_time=data[1]

    time_mask = raw_time >= 1.5
    time_masked = raw_time[time_mask]
    voltage_masked = raw_voltage[time_mask]

    sampling_rate = 1 / np.mean(np.diff(time_masked))  # Hz
    cutoff_freq = cut_off # Hz
    filter_order = 5
    nyquist_freq = 0.5 * sampling_rate
    cutoff_freq_normalized = cutoff_freq / nyquist_freq
    b, a = butter(filter_order, cutoff_freq_normalized, btype='low')
    voltage_filtered = filtfilt(b, a, voltage_masked)
    peak_index = np.argmax(voltage_filtered)

    # Smooth the data and take the derivative
    voltage_smooth = savgol_filter(voltage_filtered, window_length, polyorder)
    derivative = np.gradient(voltage_smooth)

    # Find the start of the decay using the derivative
    decay_start_index = None
    decay_thresholds = [-2, -1, -0.5,-0.3, -0.1, -0.02, -0.01, -0.005]

    for decay_threshold in decay_thresholds:
        for index in range(peak_index, len(derivative) - 1):
            if derivative[index] < decay_threshold and derivative[index + 1] < decay_threshold:
                decay_start_index = index
                break
        if decay_start_index is not None:
            break

    if decay_start_index is None:
        # Decay not found, find index closest to 1.6 seconds
        target_time = 1.6
        closest_index = min(range(len(time_masked)), key=lambda i: abs(time_masked[i] - target_time))
        decay_start_index = closest_index

    decay_start_index += 1
    time_decay = time_masked[decay_start_index:]
    time_decay = time_decay - time_decay[0]

    I_mem = np.exp((voltage_filtered[decay_start_index:])*beta)
    I_mem = I_mem - np.mean(I_mem[-500:])

    def exp_func(t, A, tau,C):
        return A* np.exp(-t / tau)+C

    # Initial guess for the parameters (A, tau)
    initial_guess = [max(I_mem),1,0.02]
    params, params_covariance = curve_fit(exp_func, time_decay, I_mem, p0=initial_guess,bounds=([0,0,-2], [np.inf,2, 2]))


    if output == 'curve':
        return I_mem, time_decay  
    elif output== 'analysis': 
        graph_time_constant_analysis(params, raw_time, raw_voltage, time_masked, voltage_filtered, derivative, decay_threshold, decay_start_index, time_decay, I_mem)   
        
    elif output == 'tc':
        return params[1]


def graph_time_constant_analysis(params_2,raw_time,raw_voltage,time_masked,voltage_filtered,derivative,decay_threshold,decay_start_index,time_decay,I_mem):
    import matplotlib.gridspec as gridspec
    
    def exp_func(t, A, tau,C):
        return A* np.exp(-t / tau)+C
    
    fig = plt.figure(figsize=(18, 8))

    # Create a gridspec object
    gs = gridspec.GridSpec(2, 6)

    ax1 = plt.subplot(gs[0, 0:2])
    ax2 = plt.subplot(gs[0, 2:4])
    ax3 = plt.subplot(gs[0, 4:6])
    ax4 = plt.subplot(gs[1, 1:3])
    ax5 = plt.subplot(gs[1, 3:5])

    # You can adjust the spacing using subplots_adjust or tight_layout as mentioned in the previous responses.
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Plot raw voltage recording
    ax1.plot(raw_time, raw_voltage)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_title('Raw Voltage Recording')

    # Plot time and filtered voltage
    ax2.plot(time_masked, voltage_filtered)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title('Reduced and Filtered Signal')

    #plot derivative
    ax3.plot(time_masked,derivative)
    ax3.plot(time_masked,decay_threshold*np.ones(len(time_masked)),'r--')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Voltage (mV)')
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_title('Derivative')


    # Plot derivative
    ax4.plot(time_decay,voltage_filtered[decay_start_index:])
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Voltage (mV)')
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_title('Isolation of decaying portion')

    # Plot log voltage
    ax5.plot(time_decay, I_mem)
    ax5.plot(time_decay, exp_func(time_decay,params_2[0],params_2[1],params_2[2]), 'b-.', label='fitted_curve:\n I_1=%5.3e, tau=%5.3f C=%5.3f' % tuple(params_2))
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('I_mem (nA)')
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.set_title('I_mem vs time fitted')
    ax5.legend(bbox_to_anchor=(.5, 1), loc='upper left',fontsize=13)
    ax5.grid(True)
    plt.text(.77, .5, r'$I_{mem}=I_{0} \times e^{V_{mem}*\beta}$', fontsize=17,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'),
         transform=plt.gca().transAxes)
    # Show the plot
    plt.show()

def measure_spike(board,input_events,duration=1,frequency=3000):
    # Define constants
    CHANNEL = 0  # Use first channel
    FREQUENCY = frequency  # Sampling frequency
    DURATION = duration# Duration of recording in seconds
    BUFFER_SIZE = int(FREQUENCY * DURATION)  # Number of samples
    try:
        # Create a DWF instance
        analog_instrument = dwf.DwfAnalogIn()
        # Open the first available device
        analog_instrument.channelEnableSet(CHANNEL, True)
        analog_instrument.channelRangeSet(CHANNEL, 1.0)  # Set voltage range to -5V to +5V
        # Set the sample rate and buffer size
        analog_instrument.frequencySet(FREQUENCY)
        analog_instrument.recordLengthSet(DURATION)
        # Wait for the device to settle
        time.sleep(.5)
        # Start the acquisition
        analog_instrument.configure(False, True)
        time.sleep(.02)
        # Start another function in a separate thread
        threading.Thread(send_virtual_events(board=board, virtual_events=input_events)).start()
        # Wait until the acquisition is done
        while True:
            if analog_instrument.status(True) == dwf.DwfStateDone:
                break
        # Get the acquired samples
        samples = analog_instrument.statusData(CHANNEL, BUFFER_SIZE)
        samples=np.array(samples)*1000
        # Generate corresponding time values
        time_values = np.linspace(0, DURATION, num=BUFFER_SIZE, endpoint=False)
    finally:
        # Always close the device
        analog_instrument.close()
        return time_values,samples
    
def measure_pulse(myConfig,model,duration=1):
    # Define constants
    CHANNEL = 0  # Use first channel
    FREQUENCY = 3000.0  # Sampling frequency
    DURATION = duration# Duration of recording in seconds
    BUFFER_SIZE = int(FREQUENCY * DURATION)  # Number of samples
    try:
        # Create a DWF instance
        analog_instrument = dwf.DwfAnalogIn()
        # Open the first available device
        analog_instrument.channelEnableSet(CHANNEL, True)
        analog_instrument.channelRangeSet(CHANNEL, 1.0)  # Set voltage range to -5V to +5V
        # Set the sample rate and buffer size
        analog_instrument.frequencySet(FREQUENCY)
        analog_instrument.recordLengthSet(DURATION)
        # Wait for the device to settle
        time.sleep(.5)
        # Start the acquisition
        analog_instrument.configure(False, True)
        time.sleep(.02)
        # Start another function in a separate thread
        threading.Thread(pulse(myConfig,model)).start()
        # Wait until the acquisition is done
        while True:
            if analog_instrument.status(True) == dwf.DwfStateDone:
                break
        # Get the acquired samples
        samples = analog_instrument.statusData(CHANNEL, BUFFER_SIZE)
        samples=np.array(samples)*1000
        # Generate corresponding time values
        time_values = np.linspace(0, DURATION, num=BUFFER_SIZE, endpoint=False)
    finally:
        # Always close the device
        analog_instrument.close()
        return time_values,samples
    
def pulse(myConfig,model):
    # set neuron latches to get DC input
    print("\nAll configurations done!\n")
    set_DC_parameter(myConfig,model,3,250)
    time.sleep(.6)
    set_DC_parameter(myConfig,model, 0, 0)
    time.sleep(3)

def Fit_FI_Curves(FF_output,neuron_config,max_val=250/250,plot=True):
    [FF_in, FF_out_PC, FF_out_PV, FF_out_SST, FF_cv] = FF_output

    window=len(FF_in)
    means_PC = np.mean(FF_out_PC, axis=0)[:window]
    stds_PC = np.std(FF_out_PC, axis=0)[:window]
    means_PV = np.mean(FF_out_PV, axis=0)[:window]
    stds_PV = np.std(FF_out_PV, axis=0)[:window]
    means_SST = np.mean(FF_out_SST, axis=0)[:window]
    stds_SST = np.std(FF_out_SST, axis=0)[:window]
    FF_in = FF_in[:window]

    # Normalize FF_in with your desired max_value (e.g., 200/250)
    normalized_FF_in = normalize_input(FF_in, max_value=max_val)

    # Calculate the fitted values
    fit_PC = custom_relu((normalized_FF_in),.06,124,60) #threshold, gain, maximum firing rate
    fit_PV = custom_relu((normalized_FF_in),.36,334,160)
    fit_SST = custom_relu((normalized_FF_in),.18,198,90)

    # Compute residuals
    residuals_PC = means_PC - fit_PC
    residuals_PV = means_PV - fit_PV
    residuals_SST = means_SST - fit_SST

    # Calculate root-mean-square error rounded to 2 decimal places

    rmse_PC = np.sqrt(np.mean(residuals_PC**2))
    rmse_PV = np.sqrt(np.mean(residuals_PV**2))
    rmse_SST = np.sqrt(np.mean(residuals_SST**2))

    if plot==True:
        #Plotting
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Plot with error bars vs input DC parameter
        ax1.errorbar(FF_in, means_PC, yerr=stds_PC, c='cornflowerblue',label='PC dynapse')
        ax1.errorbar(FF_in, means_PV, yerr=stds_PV, c='coral',label='PV dynapse')
        ax1.errorbar(FF_in, means_SST, yerr=stds_SST, c='greenyellow',label='SST dynapse')

        ax1.legend()
        ax1.set_title('F-I curve')
        ax1.set_ylabel('Output frequency (Hz)')
        ax1.set_xlabel(f'Input DC (fine parameter), coarse: {neuron_config["DC_Coarse"]}')

        # Plot the fitted curves
        ax2 = ax1.twiny()
        ax2.plot(normalized_FF_in, fit_PC, 'b-.', label='PC bio fit')
        ax2.plot(normalized_FF_in, fit_PV, 'r-.', label='PV bio fit')
        ax2.plot(normalized_FF_in, fit_SST, 'g-.', label='SST bio fit')
        ax2.set_ylim(0, 200)

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Remove right and top borders
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax2.set_xlabel('Input Pulse normalized')

        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        #legend
        ax2.legend(bbox_to_anchor=(.5, 1), loc='upper right',fontsize=13)


        #create input annotation with rmse errors rounded to two decimal places

        input_annotation = f'RMSE for PC: {np.round(rmse_PC,2)}\nRMSE for PV: {np.round(rmse_PV,2)}\nRMSE for SST: {np.round(rmse_SST,2)}'
        ax1.annotate(input_annotation, xy=(.8, 0.1), xycoords='axes fraction', size=8, bbox=dict(boxstyle="round", fc="w"))
    return rmse_PC,rmse_PV,rmse_SST


