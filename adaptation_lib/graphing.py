import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import matplotlib.pyplot as plt
import numpy as np
# Use inline backend to display the plot inline within VSCode
import matplotlib.ticker as ticker 
from adaptation_lib.spike_stats import *
from matplotlib import cm


def plot_raster_and_save(output_events,dir_path,time_label,plot_name,show=False):
    # raster plot
    raster = [[] for _ in range(max(output_events[0]) + 1)]
    for i in range(len(output_events[0])):
        raster[output_events[0][i]] += [output_events[1][i]]
    plt.figure(figsize=(10, 5))
    for k in range(len(raster)):
        plt.plot(raster[k], [k] * len(raster[k]), 'o', markersize=2)
    plt.grid(True)
    plt.ylabel('Neuron ID')
    #plt.savefig(dir_path+"/"+plot_name+"_"+time_label)
    plt.savefig(dir_path+"/"+plot_name+"_"+time_label+".svg")

    if show==True:
        plt.show()
    else:
        plt.close()



def plot_psth(test_config,spike_times, bin_size=0.011):
    """Plot the peristimulus time histogram (PSTH) for a given spike train.
    Parameters
    ----------
    spike_times : list of numpy arrays
        Spike times for each trial.
    bin_size : float
        Bin size in seconds.
    """
    plot_path=test_config['plot_path']
    time_label=test_config['time_label']
    # Calculate the PSTH
    psth, bins = psth_calc(test_config,spike_times, bin_size)
    
    # Set plot style and figure size
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    if test_config['pcn']>0:
        bar_color = (0, 0.4, 0.8, 0.5)  
    elif test_config['pvn']>0:
        bar_color = (1.00, 0.50196, 0.50196,.5)
    elif test_config['sstn']>0:
        bar_color = (1.00000, 0.65098, 0.30196,.5)
    else:
        bar_color = (0.70196, 0.70196, 0.80000,.5)
    # RGBA format (0-1 range)

    # Create the bar plot
    ax.bar(bins[:-1], psth, width=bin_size, align='edge', label='PCN', color=bar_color, edgecolor='black', linewidth=1)

    # Set axis labels and title
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Spike count', fontsize=14)
    ax.set_title('Peristimulus Time Histogram (PSTH)', fontsize=16)

    # Customize x-axis and y-axis ticks
    ax.set_xticks(np.arange(0, test_config['duration'], 0.1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # This line adjusts the y-axis ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save the plot as a high-resolution image
    plt.savefig(os.path.join(plot_path, f"psth_{time_label}.png"), dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()

def Raster_Plot(test_config, output_events, cv_values=[404,404,404], syn_values=[404,404,404], 
                save=False, show=False, save_mult=False, annotate=False, annotate_network=False):

    # Set Seaborn style
    #sns.set_style("ticks")

    # Extract parameters from test_config
    nvn, pcn, pvn, sstn = test_config['nvn'], test_config['pcn'], test_config['pvn'], test_config['sstn']
    time_label, plot_path, in_freq, date_label, test_name, in_DC = \
        test_config['time_label'], test_config['plot_path'], test_config['in_freq'], \
        test_config['date_label'], test_config['test_name'], test_config['in_DC']
    input_type = test_config['input_type']

    # Extract spike times and ids
    output_events = np.asanyarray(output_events)
    times = output_events[1] - output_events[1][0]
    spike_id = output_events[0]

    # Separate spikes by neuron type
    pc_id, pc_times = spike_id[(spike_id>nvn+1)&(spike_id<nvn+1+pcn)], times[(spike_id>nvn+1)&(spike_id<nvn+1+pcn)]
    pv_id, pv_times = spike_id[(spike_id>nvn+1+pcn)&(spike_id<nvn+2+pcn+pvn)], times[(spike_id>nvn+1+pcn)&(spike_id<nvn+2+pcn+pvn)]
    sst_id, sst_times = spike_id[(spike_id>nvn+2+pcn+pvn)&(spike_id<nvn+3+pcn+pvn+sstn)], times[(spike_id>nvn+2+pcn+pvn)&(spike_id<nvn+3+pcn+pvn+sstn)]
    input_id, input_time = spike_id[spike_id<=nvn], times[spike_id<=nvn]

    # Set input annotation
    if input_type == 'DC':
        input_annotation='Input DC: 2,'+str(in_DC)+' fine'
    else:
        input_annotation='Input Freq: '+str(in_freq)+' Hz'

    # Create plot
    plt.figure(figsize=(12,8))
    #sns.set(font_scale=1.5, style="whitegrid") # Set font scale and style
    #sns.despine() # Remove spines

    # Plot spikes
    plt.scatter(pc_times,pc_id,c='cadetblue',s=4,label='PC')
    plt.scatter(pv_times,pv_id,c='lightcoral',s=4,label='PV')
    plt.scatter(sst_times,sst_id,c='sandybrown',s=4,label='SST')
    plt.scatter(input_time,input_id,c='k',s=1,label='input')

    # Set plot parameters
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Neuron Indices', fontsize=16)
    plt.grid()
    plt.title(f"Raster Plot {test_name}", fontsize=25)
    annotation_string = f"ts: {date_label}-{time_label}"
    plt.text(0,1.023, annotation_string, transform=plt.gca().transAxes, va="top", ha="left", fontsize=6)
    # Set legend and tight layout
    plt.legend(loc="upper right", numpoints=1, fontsize=8, markerscale=3)
    plt.tight_layout()

    # Save plot
    if save_mult:
        raster_path, raster_title = test_config['raster_path'], test_config['raster_title']
        plt.savefig(f"{raster_path}/{raster_title}_{time_label}.png")
    elif save:
        plt.savefig(f"{plot_path}/Net_Raster_{time_label}.png")
    else:
        pass

    # Show or close plot
    if show:
        plt.show()
    else:
        plt.close()


def raster_plot(test_config,output_events,neuron_config,cv_values=[404,404,404],syn_values=[404,404,404],save=False,show=False,save_mult=False,annotate=False,annotate_network=False):
    nvn=test_config['nvn']
    pcn=test_config['pcn']
    pvn=test_config['pvn']
    sstn=test_config['sstn']
    time_label=test_config['time_label']
    plot_path=test_config['plot_path']
    in_freq=test_config['in_freq']
    date_label=test_config['date_label']
    test_name=test_config['test_name']
    in_DC=test_config['in_DC']

    
    output_events=np.asanyarray(output_events)
    times=output_events[1]-output_events[1][0]
    spike_id=output_events[0]

    cv_values=np.round(cv_values,2)
    syn_values=np.round(syn_values,2)

    if pcn>0:
        pc_id=spike_id[(spike_id>nvn+1)&(spike_id<nvn+1+pcn)]
        pc_times=times[(spike_id>nvn+1)&(spike_id<nvn+1+pcn)]
    else:
        pc_id=[]
        pc_times=[]

    if pvn>0:
        pv_id=spike_id[(spike_id>nvn+1+pcn)&(spike_id<nvn+2+pcn+pvn)]
        pv_times=times[(spike_id>nvn+1+pcn)&(spike_id<nvn+2+pcn+pvn)]
    else:
        pv_id=[]
        pv_times=[]

    if sstn>0:
        sst_id=spike_id[(spike_id>nvn+2+pcn+pvn)&(spike_id<nvn+3+pcn+pvn+sstn)]
        sst_times=times[(spike_id>nvn+2+pcn+pvn)&(spike_id<nvn+3+pcn+pvn+sstn)]
    else:
        sst_id=[]
        sst_times=[]

    if nvn>0:
        input_id=spike_id[spike_id<=nvn]
        input_time=times[spike_id<=nvn]
    

    if neuron_config['input_type']=='DC':
        input_annotation='Input DC: 2,'+str(in_DC)+' fine'
    else:
        input_annotation='Input Freq: '+str(in_freq)+' Hz'
       
    if annotate==True:
        plt.figure(figsize=(12,8))
        plt.subplots_adjust(bottom=0.25)
        if annotate_network==True:
            annotation_string = f"PC_gaba_gain: {neuron_config['PC_GABA_GAIN'][0]}|{neuron_config['PC_GABA_GAIN'][1]}\nPC_gaba_tau: {neuron_config['PC_GABA_TAU'][0]}|{neuron_config['PC_GABA_TAU'][1]}\nPC_nmda_gain: {neuron_config['PC_NMDA_GAIN'][0]}|{neuron_config['PC_NMDA_GAIN'][1]}\nPC_nmda_tau: {neuron_config['PC_NMDA_TAU'][0]}|{neuron_config['PC_NMDA_TAU'][1]}\nPC_shunt_gain: {neuron_config['PC_SHUNT_GAIN'][0]}|{neuron_config['PC_SHUNT_GAIN'][1]}\nPC_shunt_tau: {neuron_config['PC_SHUNT_TAU'][0]}|{neuron_config['PC_SHUNT_TAU'][1]}\nPV_inhW_PC: {neuron_config['PC_W2'][0]}|{neuron_config['PC_W2'][1]}\nSST_inhW_PC: {neuron_config['PC_W3'][0]}|{neuron_config['PC_W3'][1]}"
            plt.text(.67,-.09, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
            annotation_string = f"PV_gaba_gain: {neuron_config['PV_GABA_GAIN'][0]}|{neuron_config['PV_GABA_GAIN'][1]}\nPV_gaba_tau: {neuron_config['PV_GABA_TAU'][0]}|{neuron_config['PV_GABA_TAU'][1]}\nPV_shunt_gain: {neuron_config['PV_SHUNT_GAIN'][0]}|{neuron_config['PV_SHUNT_GAIN'][1]}\nPV_shunt_tau{neuron_config['PV_SHUNT_TAU'][0]}|{neuron_config['PV_SHUNT_TAU'][1]}\nPV_inhW_PV: {neuron_config['PV_W2'][0]}|{neuron_config['PV_W2'][1]}\nSST_inhW_PV: {neuron_config['PV_W3'][0]}|{neuron_config['PC_W3'][1]}"
            plt.text(.81,-.09, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
            annotation_string = f"SST_gaba_gain: {neuron_config['SST_GABA_GAIN'][0]}|{neuron_config['SST_GABA_GAIN'][1]}\nSST_gaba_tau: {neuron_config['SST_GABA_TAU'][0]}|{neuron_config['SST_GABA_TAU'][1]}\nSST_shunt_gain: {neuron_config['SST_SHUNT_GAIN'][0]}|{neuron_config['SST_SHUNT_GAIN'][1]},\nSST_shunt_tau{neuron_config['SST_SHUNT_TAU'][0]}|{neuron_config['SST_SHUNT_TAU'][1]}\nSST_W_PC: {neuron_config['SST_W1'][0]}|{neuron_config['SST_W1'][1]}\nSST_W_PV: {neuron_config['SST_W2'][0]}|{neuron_config['SST_W2'][1]}"
            plt.text(.95,-.09, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
        
        if pcn>0:
            annotation_string = f"PC_gain: {neuron_config['PC_GAIN'][0]}|{neuron_config['PC_GAIN'][1]}\nPC_leak: {neuron_config['PC_LEAK'][0]}|{neuron_config['PC_LEAK'][1]}\nPC_ref: {neuron_config['PC_REF'][0]}|{neuron_config['PC_REF'][1]},\nPC_spkthr: {neuron_config['PC_SPK_THR'][0]}|{neuron_config['PC_SPK_THR'][1]}\nPC_ampa_tau: {neuron_config['PC_AMPA_TAU'][0]}|{neuron_config['PC_AMPA_TAU'][1]}\nPC_ampa_gain: {neuron_config['PC_AMPA_GAIN'][0]}|{neuron_config['PC_AMPA_GAIN'][1]}\nPC_input_w: {neuron_config['PC_W0'][0]}|{neuron_config['PC_W0'][1]}\nPC_recurr_w: {neuron_config['PC_W1'][0]}|{neuron_config['PC_W1'][1]}\nPC_DC{neuron_config['PC_DC']}"
            plt.text(0,-.09, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
            annotation_string = f"PC_adaptation: {neuron_config['PC_Adaptation']}\nPCa_pwtau: {neuron_config['PC_SOAD_PWTAU_N'][0]}|{neuron_config['PC_SOAD_PWTAU_N'][1]}\nPCa_gain: {neuron_config['PC_SOAD_GAIN_P'][0]}|{neuron_config['PC_SOAD_GAIN_P'][1]}\nPCa_tau: {neuron_config['PC_SOAD_TAU_P'][0]}|{neuron_config['PC_SOAD_TAU_P'][1]},\nPCa_W: {neuron_config['PC_SOAD_W_N'][0]}|{neuron_config['PC_SOAD_W_N'][1]}\nPCa_casc: {neuron_config['PC_SOAD_CASC_P'][0]}|{neuron_config['PC_SOAD_CASC_P'][1]}"
            plt.text(.12,-.09, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
        if pvn>0:    
            annotation_string = f"PV_gain: {neuron_config['PV_GAIN'][0]}|{neuron_config['PV_GAIN'][1]}\nPV_leak: {neuron_config['PV_LEAK'][0]}|{neuron_config['PV_LEAK'][1]}\nPV_ref: {neuron_config['PV_REF'][0]}|{neuron_config['PV_REF'][1]},\nPV_spkthr: {neuron_config['PV_SPK_THR'][0]}|{neuron_config['PV_SPK_THR'][1]}\nPV_ampa_tau: {neuron_config['PV_AMPA_TAU'][0]}|{neuron_config['PV_AMPA_TAU'][1]}\nPV_ampa_gain: {neuron_config['PV_AMPA_GAIN'][0]}|{neuron_config['PV_AMPA_GAIN'][1]}\nPV_input_w: {neuron_config['PV_W0'][0]}|{neuron_config['PV_W0'][1]}\nPV_recurr_w: {neuron_config['PV_W1'][0]}|{neuron_config['PV_W1'][1]}"
            plt.text(.232,-.09, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
            annotation_string = f"Input_type: {neuron_config['input_type']}\nDC_latches: {neuron_config['DC_Latches']} \nPV_STD: {neuron_config['STD']}\nSTD_STDW_N: {neuron_config['SYAM_STDW_N'][0]}|{neuron_config['SYAM_STDW_N'][1]}\nPV_STDSTR_N: {neuron_config['SYAW_STDSTR_N'][0]}|{neuron_config['SYAW_STDSTR_N'][1]}"
            plt.text(.35,-.09, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
        if sstn>0:
            annotation_string = f"SST_gain: {neuron_config['SST_GAIN'][0]}|{neuron_config['SST_GAIN'][1]}\nSST_leak: {neuron_config['SST_LEAK'][0]}|{neuron_config['SST_LEAK'][1]}\nSST_ref: {neuron_config['SST_REF'][0]}|{neuron_config['SST_REF'][1]},\nSST_spkthr: {neuron_config['SST_SPK_THR'][0]}|{neuron_config['SST_SPK_THR'][1]}\nSST_ampa_tau: {neuron_config['SST_AMPA_TAU'][0]}|{neuron_config['SST_AMPA_TAU'][1]}\nSST_ampa_gain: {neuron_config['SST_AMPA_GAIN'][0]}|{neuron_config['SST_AMPA_GAIN'][1]}\nSST_input_w: {neuron_config['SST_W0'][0]}|{neuron_config['SST_W0'][1]}"
            plt.text(.466,-.09, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
        plt.figtext(0.847, 0.22,input_annotation, size=9,bbox=dict(boxstyle="round", fc="w"))
        annotation_string = f"CV PC: {cv_values[0]}\nCV PV: {cv_values[1]}\nCV_SST: {cv_values[2]}"
        plt.text(0.59,-.09, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
        annotation_string = f"CC PC: {syn_values[0]}\nCC PV: {syn_values[1]}\nCC_SST: {syn_values[2]}"
        plt.text(0.59,-.16, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
    else:
        plt.figure(figsize=(10,8))
        plt.figtext(0.8, 0.13,input_annotation, size=8,bbox=dict(boxstyle="round", fc="w"))
        annotation_string = f"CV_PC: {cv_values[0]}\nCV_PV: {cv_values[1]}\nCV_SST: {cv_values[2]}"
        plt.text(.706,-.07, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
        annotation_string = f"CC_PC: {syn_values[0]}\nCC_PV: {syn_values[1]}\nCC_SST: {syn_values[2]}"
        plt.text(.806,-.07, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8,bbox=dict(boxstyle="round", fc="w"))
    SMALL_SIZE = 15
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE-5)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE) 
    plt.scatter(pc_times,pc_id,c='cadetblue',s=4,label='PC')
    plt.scatter(pv_times,pv_id,c='lightcoral',s=4,label='PV')
    plt.scatter(sst_times,sst_id,c='sandybrown',s=4,label='SST')
    plt.scatter(input_time,input_id,c='k',s=1,label='input')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Indicies')
    plt.grid()
    plt.title(f"Raster Plot {test_name}",fontsize=25)
    annotation_string = f"ts: {date_label}-{time_label}"
    plt.text(0,1.023, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=6)
    plt.legend(loc="upper right", numpoints=1, fontsize=8,markerscale=3)
    plt.tight_layout()

    if save_mult==True:
        raster_path=test_config['raster_path']
        raster_title=test_config['raster_title']
        plt.savefig(raster_path+"/"+raster_title+"_"+time_label+".png")
    elif save==True:
        plt.savefig(plot_path+"/Net_Raster_"+time_label+".png")
    else:
        pass
    if show==True:
        plt.show()
    else:
        plt.close()

##############
# Frequency over time graphs
##############

def mean_frequency_vs_time_graph(ff_output, test_config, neuron_config, annotate=False, annotate_network=False):
    neuron_types = ['PC', 'PV', 'SST']
    colors = {'PC': 'b', 'PV': 'r', 'SST': 'orange'}
    alphas = {'PC': 0.2, 'PV': 0.2, 'SST': 0.2}
    neuron_counts = {nt: test_config[nt.lower() + 'n'] for nt in neuron_types}
    graph_type=1
    fig, ax = plt.subplots(figsize=(12,8) if annotate else (10,8))

    for nt in neuron_types:
        if neuron_counts[nt] > 0:
            if graph_type==0:
                means = np.mean(ff_output[neuron_types.index(nt) + 1], axis=0)
                stds = np.std(ff_output[neuron_types.index(nt) + 1], axis=0)
                ax.plot(ff_output[0], means, c=colors[nt], label=nt)
                #plot with error bars
                ax.errorbar(ff_output[0], means, yerr=stds, c=colors[nt])
                #ax.fill_between(ff_output[0], means - stds, means + stds, color=colors[nt], alpha=alphas[nt])
            elif graph_type==1:
                for i in range(len(ff_output[neuron_types.index(nt) + 1])):
                    ax.plot(ff_output[0], ff_output[neuron_types.index(nt) + 1][i], c=colors[nt], alpha=.1)
                means = np.mean(ff_output[neuron_types.index(nt) + 1], axis=0)
                ax.plot(ff_output[0], means, c=colors[nt], label=nt)




    if annotate:
        plt.subplots_adjust(bottom=0.25)
        annotation_box_config = dict(boxstyle="round", fc="w")
        for i, nt in enumerate(neuron_types):
            if neuron_counts[nt] > 0:
                annotation_string = f"{nt}_gain: {neuron_config[nt + '_GAIN'][0]}|{neuron_config[nt + '_GAIN'][1]}\n{nt}_leak: {neuron_config[nt + '_LEAK'][0]}|{neuron_config[nt + '_LEAK'][1]}"
                annotation_string += f"\n{nt}_ref: {neuron_config[nt + '_REF'][0]}|{neuron_config[nt + '_REF'][1]}\n{nt}_spk_thr: {neuron_config[nt + '_SPK_THR'][0]}|{neuron_config[nt + '_SPK_THR'][1]}"
                annotation_string += f"\n{nt}_ampa_tau: {neuron_config[nt + '_AMPA_TAU'][0]}|{neuron_config[nt + '_AMPA_TAU'][1]}\n{nt}_ampa_gain: {neuron_config[nt + '_AMPA_GAIN'][0]}|{neuron_config[nt + '_AMPA_GAIN'][1]}"
                annotation_string += f"\n{nt}_W_0: {neuron_config[nt + '_W0'][0]}|{neuron_config[nt + '_W0'][1]}\n{nt}_W_1: {neuron_config[nt + '_W1'][0]}|{neuron_config[nt + '_W1'][1]}"
                annotation_string += f"\n{nt}_W_2: {neuron_config[nt + '_W2'][0]}|{neuron_config[nt + '_W2'][1]}\n{nt}_W_3: {neuron_config[nt + '_W3'][0]}|{neuron_config[nt + '_W3'][1]}"
                annotation_string += f"\n{nt}_DC: {neuron_config[nt + '_DC']}"
                if nt == 'PC':
                    annotation_string_2 = f"\n{nt}_adaptation: {neuron_config[nt + '_Adaptation']}"
                    annotation_string_2 += f"\n{nt}_pwtau: {neuron_config[nt + '_SOAD_PWTAU_N'][0]}|{neuron_config[nt + '_SOAD_PWTAU_N'][1]}"
                    annotation_string_2 += f"\n{nt}_gain: {neuron_config[nt + '_SOAD_GAIN_P'][0]}|{neuron_config[nt + '_SOAD_GAIN_P'][1]}"
                    annotation_string_2 += f"\n{nt}_tau: {neuron_config[nt + '_SOAD_TAU_P'][0]}|{neuron_config[nt + '_SOAD_TAU_P'][1]}"
                    annotation_string_2 += f"\n{nt}_w: {neuron_config[nt + '_SOAD_W_N'][0]}|{neuron_config[nt + '_SOAD_W_N'][1]}"
                    annotation_string_2 += f"\n{nt}_casc: {neuron_config[nt + '_SOAD_CASC_P'][0]}|{neuron_config[nt + '_SOAD_CASC_P'][1]}"
                    plt.text(.12*5,-.09,annotation_string_2, transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox=annotation_box_config)
                if nt == 'PV': 
                    annotation_string_3 = f"\n{nt}_STD: {neuron_config['STD']}"
                    annotation_string_3 += f"\n{nt}_: {neuron_config['SYAM_STDW_N'][0]}|{neuron_config['SYAM_STDW_N'][1]}"
                    annotation_string_3 += f"\n{nt}_: {neuron_config['SYAW_STDSTR_N'][0]}|{neuron_config['SYAW_STDSTR_N'][1]}"
                    plt.text(.12*6,-.09,annotation_string_3, transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox=annotation_box_config)

                plt.text(.12*i, -.09, annotation_string, transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox=annotation_box_config)

    # Further annotations and text can be added similarly

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Output frequency (Hz)')

    if neuron_config['input_type']=='DC':
        ax.set_xlabel('DC fine value')
    else:
        ax.set_xlabel('Input Frequency (Hz)')

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    
    ax.set_title(test_config['test_name'])
    ax.legend()
    fig.tight_layout()
    plt.savefig(test_config['plot_path']+"/"+test_config['time_label'], bbox_inches="tight")
    plt.show()

def frequency_vs_time_graph(fot_output,test_config,save=False,show=False):
    '''I am a docstring'''
    plt.figure()
    plt.plot(fot_output[0],fot_output[1][0:len(fot_output[0])],c='cadetblue',label=' PC_neurons')
    plt.plot(fot_output[0],fot_output[2][0:len(fot_output[0])],c='lightcoral',label=' PV_neurons')
    plt.plot(fot_output[0],fot_output[3][0:len(fot_output[0])],c='sandybrown',label=' SST_neurons')
    plt.xlabel('Time (s)')
    plt.ylabel('Output Frequency (Hz)')
    plt.title(f'Frequeny vs Time')
    plt.figtext(0.7, 0.15,'Input Freq: '+str(test_config['in_freq'])+' Hz', size=9,bbox=dict(boxstyle="round", fc="w"))
    plt.grid()
    plt.legend()
    if save==True:
        plt.savefig(test_config['plot_path']+"/FvT_"+test_config['time_label']+".svg")
    else:
        pass
    if show==True:
        plt.show()
    else:
        plt.close()

def frequency_vs_time_graph_sweep(ffdata,neuron_config,test_config):
    """
    Plots the firing frequency data for PC, PV, and SST tests from given ffdata.
    Separate plots are created for each test type if there is actual data for that test.
    """
    
    # Set Seaborn style in Matplotlib
    plt.style.use('seaborn')
    labels=neuron_config['sweep_range_fine']

    # Define a list of colors for the error bars
    n_colors = len(ffdata)
    colors = plt.cm.tab10(np.linspace(0, 1, n_colors))

    if test_config['pcn']>0:
        # Plot PC data
        plt.figure(figsize=(12,8))
        for i, ff_output in enumerate(ffdata):
            FF_in, FF_out_PC = ff_output[0], ff_output[1]
            mean_PC = np.mean(FF_out_PC, axis=0)
            std_PC = np.std(FF_out_PC, axis=0)
            plt.errorbar(FF_in, mean_PC, yerr=std_PC, label=int(labels[i]), alpha=0.8,color=colors[i])
        plt.legend(title='Fine_Value', facecolor='white', framealpha=1)
        if neuron_config['input_type']=='DC':
            plt.xlabel("DC Fine Value")
        else:
            plt.xlabel("Input Frequency (Hz)")
        plt.ylabel("Firing Frequency (Hz)")
        plt.ylim(0, 400)
        annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
        plt.text(0,1.023, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=6)
        plt.title("FF PC Sweep: "+str(neuron_config['sweep_variable'])+" Coarse: "+str(neuron_config['sweep_coarse_val']))
        plt.savefig(test_config['plot_path']+"/"+test_config['time_label'],bbox_inches="tight")
        plt.show()
    

    if test_config['pvn']>0:
        # Plot PV data
        plt.figure(figsize=(12,8))
        for i, ff_output in enumerate(ffdata):
            FF_in, FF_out_PC = ff_output[0], ff_output[2]
            mean_PC = np.mean(FF_out_PC, axis=0)
            std_PC = np.std(FF_out_PC, axis=0)
            plt.errorbar(FF_in, mean_PC, yerr=std_PC, label=int(labels[i]), alpha=0.8,color=colors[i])
        plt.legend(title='Fine_Value', facecolor='white', framealpha=1)
        if neuron_config['input_type']=='DC':
            plt.xlabel("DC Fine Value")
        else:
            plt.xlabel("Input Frequency (Hz)")
        plt.ylabel("Firing Frequency (Hz)")
        annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
        plt.text(0,1.023, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=6)
        plt.title("FF PV Sweep: "+str(neuron_config['sweep_variable'])+" Coarse: "+str(neuron_config['sweep_coarse_val']))
        plt.savefig(test_config['plot_path']+"/"+test_config['time_label'],bbox_inches="tight")
        plt.show()
    

    if test_config['sstn']>0:
        # Plot SST data
        plt.figure(figsize=(12,8))
        for i, ff_output in enumerate(ffdata):
            FF_in, FF_out_PC = ff_output[0], ff_output[3]
            mean_PC = np.mean(FF_out_PC, axis=0)
            std_PC = np.std(FF_out_PC, axis=0)
            plt.errorbar(FF_in, mean_PC, yerr=std_PC, label=int(labels[i]), alpha=0.8,color=colors[i])
        plt.legend(title='Fine_Value', facecolor='white', framealpha=1)
        if neuron_config['input_type']=='DC':
            plt.xlabel("DC Fine Value")
        else:
            plt.xlabel("Input Frequency (Hz)")
        plt.ylabel("Firing Frequency (Hz)")
        annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
        plt.text(0,1.023, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=6)
        plt.title("FF SST Sweep: "+str(neuron_config['sweep_variable'])+" Coarse: "+str(neuron_config['sweep_coarse_val']))
        plt.savefig(test_config['plot_path']+"/"+test_config['time_label'],bbox_inches="tight")
        plt.show()
    '''''
    if test_config['tcun']>0:
        # Plot SST data
        plt.figure(figsize=(12,8))
        for i, ff_output in enumerate(ffdata):
            FF_in, FF_out_PC = ff_output[0], ff_output[3]
            mean_PC = np.mean(FF_out_PC, axis=0)
            std_PC = np.std(FF_out_PC, axis=0)
            plt.errorbar(FF_in, mean_PC, yerr=std_PC, label=int(labels[i]), alpha=0.8,color=colors[i])
        plt.legend(title='Fine_Value', facecolor='white', framealpha=1)
        if neuron_config['input_type']=='DC':
            plt.xlabel("DC Fine Value")
        else:
            plt.xlabel("Input Frequency (Hz)")
        plt.ylabel("Firing Frequency (Hz)")
        annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
        plt.text(0,1.023, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=6)
        plt.title("FF TCU Sweep: "+str(neuron_config['sweep_variable'])+" Coarse: "+str(neuron_config['sweep_coarse_val']))
        plt.savefig(test_config['plot_path']+"/"+test_config['time_label'],bbox_inches="tight")
        plt.show()
    '''''

def Sweep_rate_graph(sweep_rates_output,sweep_variable,sweep_range,test_config):
        pcn=test_config['pcn']
        pvn=test_config['pvn']
        sstn=test_config['sstn']
        test_name=test_config['test_name']
        coarse=test_config['coarse']
        sweep_in=sweep_range
        sweep_out_PC=[result[0] for result in sweep_rates_output]
        sweep_out_PV=[result[1] for result in sweep_rates_output]
        sweep_out_SST=[result[2] for result in sweep_rates_output]
        time_label=test_config['time_label']
        plot_path=test_config['plot_path']
        in_freq=test_config['in_freq']

        if pcn>0:
            mean_PC=np.mean(sweep_out_PC,axis=1)
            std_PC=np.std(sweep_out_PC,axis=1)
            plt.plot(sweep_in,mean_PC,c='b',label='PC')
            plt.fill_between(sweep_in, mean_PC - std_PC, mean_PC + std_PC,color='cadetblue', alpha=0.3)
        if pvn>0:
            mean_PV=np.mean(sweep_out_PV,axis=1)
            std_PV=np.std(sweep_out_PV,axis=1)
            plt.plot(sweep_in,mean_PV,c='r',label='PV')
            plt.fill_between(sweep_in, mean_PV - std_PV, mean_PV + std_PV,color='lightcoral', alpha=0.2)
        if sstn>0:
            mean_SST=np.mean(sweep_out_SST,axis=1)
            std_SST=np.std(sweep_out_SST,axis=1)
            plt.plot(sweep_in,mean_SST,c='orange',label='SST')
            plt.fill_between(sweep_in, mean_SST - std_SST, mean_SST + std_SST,color='sandybrown', alpha=0.2)
        
        input_annotation='Input Freq: '+str(in_freq)+' Hz'
        plt.figtext(0.8, 0.13,input_annotation, size=8,bbox=dict(boxstyle="round", fc="w"))
        annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
        plt.text(0,1.023, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=6)
        plt.ylabel('Output frequency (Hz)')
        plt.title('Paramater Sweep '+str(test_name))
        plt.xlabel(sweep_variable+'_'+str([coarse,'X']))
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_path+"/"+time_label,bbox_inches="tight")
        plt.show()

def Sweep_FOT_graph(sweep_rates_output,sweep_variable,sweep_range,sweep_coarse_val,test_config):
        time_label=test_config['time_label']
        plot_path=test_config['plot_path']
        in_freq=test_config['in_freq']
        input_annotation='Input Freq: '+str(in_freq)+' Hz'
        annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
        pcn=test_config['pcn']
        pvn=test_config['pvn']
        sstn=test_config['sstn']
        test_name=test_config['test_name']
   
    
        if pcn>0:
            plt.figure()
            # Apply Seaborn's "whitegrid" style settings to your plot
            plt.style.use('seaborn')
            n_lines = len(sweep_rates_output)
            # Create a color palette
            colors = cm.Blues(np.linspace(0.3, 1, n_lines))

            for i in range(n_lines):
                fot_output = sweep_rates_output[i]

                # Use a blue color from the palette for every line
                line_color = colors[i]

                plt.plot(fot_output[0], fot_output[1][0:len(fot_output[0])], label=''+str([sweep_coarse_val, sweep_range[i]]), color=line_color)

            plt.xlabel('Time (s)')
            plt.ylabel('Output Frequency (Hz)')
            plt.title(f'Frequeny over Time | PC')
            plt.figtext(0.7, 0.15,'Input Freq: '+str(test_config['in_freq'])+' Hz', size=9,bbox=dict(boxstyle="round", fc="w"))
            plt.grid()
            legend = plt.legend(loc="upper right", title=sweep_variable, numpoints=1, fontsize=8, markerscale=3, title_fontsize=8, frameon=True)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_boxstyle('round', rounding_size=0.1)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path+"/FVT"+time_label,bbox_inches="tight")
            plt.text(0,1.025, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8)
            plt.show()
        if pvn>0:
            plt.figure()
            # Apply Seaborn's "whitegrid" style settings to your plot
            plt.style.use('seaborn')
            n_lines = len(sweep_rates_output)
            # Create a color palette
            colors = cm.Reds(np.linspace(0.3, 1, n_lines))

            for i in range(len(sweep_rates_output)):
                fot_output=sweep_rates_output[i]
                # Use a red color from the palette for every line
                line_color = colors[i]

                plt.plot(fot_output[0],fot_output[2][0:len(fot_output[0])],color=line_color,label=''+str([sweep_coarse_val,sweep_range[i]]))
            plt.xlabel('Time (s)')
            plt.ylabel('Output Frequency (Hz)')
            plt.title(f'Frequeny vs Time')
            plt.figtext(0.7, 0.15,'Input Freq: '+str(test_config['in_freq'])+' Hz', size=9,bbox=dict(boxstyle="round", fc="w"))
            plt.grid()
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path+"/FVT"+time_label,bbox_inches="tight")
            plt.text(0,1.025, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8)
            plt.show()
        if sstn>0:
            plt.figure()
            # Apply Seaborn's "whitegrid" style settings to your plot
            plt.style.use('seaborn')
            n_lines = len(sweep_rates_output)
            # Create a color palette
            colors = cm.Orange(np.linspace(0.3, 1, n_lines))

            for i in range(len(sweep_rates_output)):
                fot_output=sweep_rates_output[i]
                # Use a orange color from the palette for every line
                line_color = colors[i]

                plt.plot(fot_output[0],fot_output[3][0:len(fot_output[0])],color=line_color,label=''+str([sweep_coarse_val,sweep_range[i]]))
            plt.xlabel('Time (s)')
            plt.ylabel('Output Frequency (Hz)')
            plt.title(f'Frequeny vs Time')
            plt.figtext(0.7, 0.15,'Input Freq: '+str(test_config['in_freq'])+' Hz', size=9,bbox=dict(boxstyle="round", fc="w"))
            plt.grid()
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path+"/FVT"+time_label,bbox_inches="tight")
            plt.text(0,1.025, annotation_string,transform=plt.gca().transAxes, va = "top", ha="left",fontsize=8)
            plt.show()



        

def overtake_graph(x,y,neuron_config,test_config):
        time_label=test_config['time_label']
        plot_path=test_config['plot_path']
        mean=np.mean(y,axis=0)
        print(mean)
        std=np.std(y,axis=0)
        print(std)
        plt.plot(x,mean,c='b',label='PC')
        plt.fill_between(x, mean - std, mean + std,color='magenta', alpha=0.3)
        plt.ylabel('Adaptation Time (s)')
        plt.title('Adaptation Time: '+neuron_config['input_type']+' input')
        plt.xlabel('Frequency (Hz)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path+"/overtake_"+time_label,bbox_inches="tight")
        plt.show()

def decay_grap(x,y,sweep_variable,test_config):
        coarse=test_config['coarse']
        y=np.array(y,dtype=object)
        if any(y[:,0])>0:
            plt.plot(x,y[:,0],color='cadetblue',label='PC')
        if any(y[:,1])>0:
            plt.plot(x,y[:,1],color='lightcoral',label='PV')
        if any(y[:,2])>0:
            plt.plot(x,y[:,2],color='sandybrown',label='SST')
        plt.ylabel('Decay times')
        plt.title('Decay Time: '+test_config['test_name']+test_config['input_type'])
        plt.xlabel(sweep_variable+'_'+str([coarse,'X']))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(test_config['plot_path']+"/decay"+test_config['time_label'],bbox_inches="tight")
        plt.show()
