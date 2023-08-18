import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import matplotlib.pyplot as plt
import numpy as np
# Use inline backend to display the plot inline within VSCode
import matplotlib.ticker as ticker 
from adaptation_lib.spike_stats import *
from matplotlib import cm

##############
# Raster Plots
##############

def simple_raster_plot(output_events,dir_path,time_label,plot_name,show=False):
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

def script_annotated_raster_plot(test_config,output_events,neuron_config,cv_values=[404,404,404],syn_values=[404,404,404],save=False,save_mult=False,annotate=False,annotate_network=False):
    neuron_types = ['PC', 'PV', 'SST']
    colors = {'PC': 'b', 'PV': 'r', 'SST': 'orange'}
    neuron_counts = {nt: test_config[nt.lower() + 'n'] for nt in neuron_types}
    test_config_keys = ['nvn', 'pcn', 'pvn', 'sstn', 'time_label', 'plot_path', 'in_freq', 'in_DC']
    nvn, pcn, pvn, sstn, time_label, plot_path, in_freq, in_DC = (test_config[key] for key in test_config_keys)
    plt.style.use('seaborn-white')
    output_events=np.asanyarray(output_events)
    times=output_events[1]-output_events[1][0]
    spike_id=output_events[0]

    cv_values=np.round(cv_values,2)
    syn_values=np.round(syn_values,2)

    pc_id, pc_times = get_id_times(nvn, pcn, spike_id, times)
    pv_id, pv_times = get_id_times(nvn + 1 + pcn, pvn, spike_id, times)
    sst_id, sst_times = get_id_times(nvn + 2 + pcn + pvn, sstn, spike_id, times)

    if nvn > 0:
        input_id = spike_id[spike_id <= nvn]
        input_time = times[spike_id <= nvn]
   
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(pc_times,pc_id,c='cadetblue',s=4,label='PC')
    ax.scatter(pv_times,pv_id,c='lightcoral',s=4,label='PV')
    ax.scatter(sst_times,sst_id,c='sandybrown',s=4,label='SST')
    ax.scatter(input_time,input_id,c='k',s=1,label='input')
    ax.set_xlabel('Time (s)',fontsize=18)
    ax.set_ylabel('Neuron Indicies',fontsize=18)
    ax.set_title(f"Raster Plot {test_config['test_name']}", fontsize=20)
    ax.legend(loc="upper right", title='Neruton Type', numpoints=1, fontsize=8, markerscale=3, title_fontsize=8, frameon=True)
    #set Time stamp
    annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
    ax.text(0, 1.023, annotation_string, size =12, transform=ax.transAxes, va = "top", ha="left")
    #set input strenght annotation
    input_annotation = f'Input DC: 3,{in_DC} fine' if neuron_config['input_type'] == 'DC' else f'Input Freq: {in_freq} Hz'
    ax.annotate(input_annotation, xy=(0.9, -0.015), xycoords='axes fraction', size=8, bbox=dict(boxstyle="round", fc="w"))
    #set CV and Synaptic strength annotation
    if cv_values[0]!=404:
        ax.annotate(f'CV: {cv_values[0]}|{cv_values[1]}|{cv_values[2]}',  xy=(0.9, -0.012), xycoords='axes fraction',size=8, bbox=dict(boxstyle="round", fc="w"))
    if syn_values[0]!=404:
        ax.annotate(f'Syn: {syn_values[0]}|{syn_values[1]}|{syn_values[2]}', xy=(0.9, -0.09), xycoords='axes fraction' ,size=8, bbox=dict(boxstyle="round", fc="w"))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if annotate==True:
        annotate_plot(neuron_types,neuron_counts,neuron_config,ax)
    if annotate_network==True: 
        annotate_plot_network(neuron_types,neuron_counts,neuron_config,ax)

    plt.tight_layout()

    if save_mult==True:
        raster_path=test_config['raster_path']
        raster_title=test_config['raster_title']
        plt.savefig(raster_path+"/"+raster_title+"_"+time_label+".svg")
    elif save==True:
        plt.savefig(plot_path+"/Net_Raster_"+time_label+".svg")
    else:
        pass

    plt.close()
    
    return fig

def get_id_times(nvn, pcn, test_id, time):
    id = []
    times = []
    if pcn > 0:
        condition = (test_id > nvn + 1) & (test_id < nvn + 1 + pcn)
        id = test_id[condition]
        times = time[condition]
    return id, times


##############
# Frequency output over 1 second vs various inputs
##############

def frequency_vs_input_plot(ff_output, test_config,neuron_config, annotate=False):
    #Frequency vs input, plots out the means of each populations type 
    neuron_types = ['PC', 'PV', 'SST']
    colors = {'PC': 'b', 'PV': 'r', 'SST': 'orange'}
    neuron_counts = {nt: test_config[nt.lower() + 'n'] for nt in neuron_types}
    graph_type=0
    plt.style.use('seaborn-white')
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

    # Further annotations and text can be added similarly
    if annotate==True:
        annotate_plot(neuron_types,neuron_counts,neuron_config,ax)
    #set input strenght annotation
    input_annotation = f'Input type: {neuron_config["input_type"]}'
    ax.annotate(input_annotation, xy=(0.9, -0.09), xycoords='axes fraction', size=8, bbox=dict(boxstyle="round", fc="w"))
    #set Time stamp
    annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
    ax.text(0, 1.023, annotation_string, size =12, transform=ax.transAxes, va = "top", ha="left")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Output frequency (Hz)',fontsize=18)
    ax.set_xlabel("DC Fine Value" if test_config['input_type']=='DC' else "Input Frequency (Hz)",fontsize=18)
    ax.set_title(f" {test_config['test_name']}", fontsize=20)
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title(test_config['test_name'])
    ax.legend()
    fig.tight_layout()
    plt.savefig(test_config['plot_path']+"/"+test_config['time_label'], bbox_inches="tight")
    plt.show()

def frequency_vs_input_simple(ff_output,neuron_config, annotate=False):
    #Frequency vs input, plots out the means of each populations type 
    neuron_types = ['PC', 'PV', 'SST']
    colors = {'PC': 'b', 'PV': 'r', 'SST': 'orange'}
    graph_type=0
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(figsize=(12,8) if annotate else (10,8))

    for nt in neuron_types:
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

    # Further annotations and text can be added similarly
    #set input strenght annotation
    input_annotation = f'Input type: {neuron_config["input_type"]}'
    ax.annotate(input_annotation, xy=(0.9, -0.09), xycoords='axes fraction', size=8, bbox=dict(boxstyle="round", fc="w"))
    #set Time stamp
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Output frequency (Hz)',fontsize=18)
    ax.set_xlabel("DC Fine Value" if neuron_config['input_type']=='DC' else "Input Frequency (Hz)",fontsize=18)
    ax.set_title(f"STUFF", fontsize=20)
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend()
    fig.tight_layout()
    plt.show()



def sweep_frequency_vs_input_plot(ffdata, test_config):
    """
    Plots the firing frequency data for PC, PV, and SST tests from given ffdata.
    """
    # Set Seaborn white style in Matplotlib
    plt.style.use('seaborn-white')
    labels = test_config['sweep_range_fine']

    # Define a list of colors for the error bars
    n_colors = len(ffdata)
    colors = plt.cm.tab10(np.linspace(0, 1, n_colors))

    test_types = {
        'pcn': [0, 'b', 'cadetblue', 'PC'],
        'pvn': [1, 'r', 'lightcoral', 'PV'],
        'sstn': [2, 'orange', 'sandybrown', 'SST']
    }

    fig, ax = plt.subplots(figsize=(12,8))

    for test_type, value in test_types.items():
        if test_config[test_type] > 0:
            print('flag 1')
            print(test_type)
            for i, ff_output in enumerate(ffdata):
                FF_in, FF_out = ff_output[0], ff_output[value[0]]
                mean_FF = np.mean(FF_out, axis=0)
                std_FF = np.std(FF_out, axis=0)
                ax.errorbar(FF_in, mean_FF, yerr=std_FF, label=value[1] + ' ' + str(labels[i]), alpha=0.8, color=colors[i])

    ax.legend(loc="upper right", title=test_config['sweep_variable'], numpoints=1, fontsize=8, markerscale=3, title_fontsize=8, frameon=True)
    ax.set_xlabel("DC Fine Value" if test_config['input_type']=='DC' else "Input Frequency (Hz)")
    ax.set_ylabel("Firing Frequency (Hz)")
    annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
    ax.text(0, 1.023, annotation_string, transform=ax.transAxes, va="top", ha="left", fontsize=12)
    ax.set_ylim(0, 400)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(f"{test_config['plot_path']}/{test_config['time_label']}", bbox_inches="tight")
    plt.show()



##########
#Frequency output over 1 second vs a sweept parameter plot
##########
def sweep_frequency_vs_parameter_plot(sweep_rates_output, test_config):
    """
    Plots the firing frequency data for PC, PV, and SST tests from given sweep_rates_output.
    """
    # Set Seaborn white style in Matplotlib
    plt.style.use('seaborn-white')

    test_types = {
        'pcn': [0, 'b', 'cadetblue', 'PC'],
        'pvn': [1, 'r', 'lightcoral', 'PV'],
        'sstn': [2, 'orange', 'sandybrown', 'SST']
    }

    fig, ax = plt.subplots()

    for test_type, value in test_types.items():
        if test_config[test_type] > 0:
            sweep_out = [result[value[0]] for result in sweep_rates_output]
            mean_out = np.mean(sweep_out, axis=1)
            std_out = np.std(sweep_out, axis=1)

           
            ax.errorbar(test_config['sweep_range_fine'], mean_out, yerr=std_out,c=value[1], label=value[3])
            '''
            ax.plot(test_config['sweep_range_fine'], mean_out, c=value[1], label=value[3])
            ax.fill_between(test_config['sweep_range_fine'], mean_out - std_out, mean_out + std_out, color=value[2], alpha=0.3)
            '''
    #set input strenght annotation
    input_annotation = f'Input DC: 3,{test_config["in_DC"]} fine' if test_config['input_type'] == 'DC' else f'Input Freq: {test_config["in_freq"]} Hz'
    ax.annotate(input_annotation, xy=(0.9, -0.035), xycoords='axes fraction', size=8, bbox=dict(boxstyle="round", fc="w"))
    timestamp_annotation = f"ts: {test_config['date_label']}-{test_config['time_label']}"
    ax.text(0, 1.023, timestamp_annotation, transform=ax.transAxes, va = "top", ha="left", fontsize=8)
    ax.set_ylabel('Output frequency (Hz)')
    ax.set_title(f'Paramater Sweep {test_config["test_name"]}')
    ax.set_xlabel(f'{[test_config["sweep_variable"]]}_{[test_config["coarse"], "X"]}')
    ax.legend()

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(f"{test_config['plot_path']}/{test_config['time_label']}", bbox_inches="tight")
    plt.show()


##########
#Frequency over a time bin vs time plots
##########

def sweep_frequency_vs_time_plot(sweep_rates_output, test_config):
    """
    Plots the firing frequency data for PC, PV, and SST tests from given sweep_rates_output.
    """
    sweep_range=test_config['sweep_range_fine']
    sweep_coarse_val=test_config['sweep_coarse_val']
    # Set Seaborn white style in Matplotlib
    plt.style.use('seaborn-white')

    test_types = {
        'pcn': [1, "Frequeny over Time | PC", cm.Blues],
        'pvn': [2, "Frequeny over Time | PV", cm.Reds],
        'sstn': [3, "Frequeny over Time | SST", cm.Oranges]
    }

    fig, ax = plt.subplots()

    for test_type, value in test_types.items():
        if test_config[test_type] > 0:
            n_lines = len(sweep_rates_output)
            colors = value[2](np.linspace(0.3, 1, n_lines))

            for i in range(n_lines):
                fot_output = sweep_rates_output[i]
                ax.plot(fot_output[0], fot_output[value[0]][0:len(fot_output[0])], label=''+str([sweep_coarse_val, sweep_range[i]]), color=colors[i])
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Output Frequency (Hz)')
            ax.set_title(value[1])
            ax.legend(loc="upper right", title=test_config['sweep_variable'], numpoints=1, fontsize=8, markerscale=3, title_fontsize=8, frameon=True)
            ax.text(0.7, 0.15, 'Input Freq: '+str(test_config['in_freq'])+' Hz', size=9, bbox=dict(boxstyle="round", fc="w"))
            annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
            ax.text(0,1.025, annotation_string, transform=ax.transAxes, va = "top", ha="left",fontsize=8)

    #set input strenght annotation
    input_annotation = f'Input DC: 3,{test_config["in_DC"]} fine' if test_config['input_type'] == 'DC' else f'Input Freq: {test_config["in_freq"]} Hz'
    ax.annotate(input_annotation, xy=(0.9, -0.035), xycoords='axes fraction', size=8, bbox=dict(boxstyle="round", fc="w"))
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.savefig(f"{test_config['plot_path']}/FVT{test_config['time_label']}.svg", bbox_inches="tight")
    plt.show()

def frequency_vs_time_plot(fot_output,test_config,save=False,annotate=False,show=False):
    '''I am a docstring'''
    plt.style.use('seaborn-white')

    fig, ax = plt.subplots(figsize=(12,8) if annotate else (12,8))
    ax.plot(fot_output[0],fot_output[1][0:len(fot_output[0])],c='cadetblue',label=' PC_neurons')
    ax.plot(fot_output[0],fot_output[2][0:len(fot_output[0])],c='lightcoral',label=' PV_neurons')
    ax.plot(fot_output[0],fot_output[3][0:len(fot_output[0])],c='sandybrown',label=' SST_neurons')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output Frequency (Hz)')
    ax.set_title(f'Frequeny vs Time')
    plt.figtext(0.7, 0.15,'Input Freq: '+str(test_config['in_freq'])+' Hz', size=9,bbox=dict(boxstyle="round", fc="w"))
    ax.legend()
    annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
    ax.text(0, 1.023, annotation_string, size =12, transform=ax.transAxes, va = "top", ha="left")
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if save==True:
        plt.savefig(test_config['plot_path']+"/FvT_"+test_config['time_label']+".png")
    else:
        pass
    plt.close()
    
    return fig
    

################
# PSTH plots
################

def plot_psth(test_config,spike_times, bin_size=0.011):
    test_config_keys = ['nvn', 'pcn', 'pvn', 'sstn', 'time_label', 'plot_path', 'in_freq', 'in_DC']
    nvn, pcn, pvn, sstn, time_label, plot_path, in_freq, in_DC = (test_config[key] for key in test_config_keys)
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
    plt.style.use('seaborn-white')
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
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #set Time stamp
    annotation_string = f"ts: {test_config['date_label']}-{test_config['time_label']}"
    ax.text(0, 1.023, annotation_string, size =12, transform=ax.transAxes, va = "top", ha="left")
    #set input strenght annotation
    input_annotation = f'Input DC: 3,{in_DC} fine' if test_config['input_type'] == 'DC' else f'Input Freq: {in_freq} Hz'
    ax.annotate(input_annotation, xy=(0.9, -0.03), xycoords='axes fraction', size=8, bbox=dict(boxstyle="round", fc="w"))
    # Customize x-axis and y-axis ticks
    ax.set_xticks(np.arange(0, test_config['duration'], 0.1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # This line adjusts the y-axis ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save the plot as a high-resolution image
    plt.savefig(os.path.join(plot_path, f"psth_{time_label}.png"), dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()

################
# Interaction in between neuron plots
################

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

################
# Decay graph
################

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

###############
#Graph Annotation
###############

def annotate_plot(neuron_types,neuron_counts,neuron_config,ax):

        for i, nt in enumerate(neuron_types):
            if neuron_counts[nt] > 0:
                annotation_string = f"\n{nt}_W_0: {neuron_config[nt + '_W0'][0]}|{neuron_config[nt + '_W0'][1]}\n{nt}_W_1: {neuron_config[nt + '_W1'][0]}|{neuron_config[nt + '_W1'][1]}"
                annotation_string += f"\n{nt}_W_2: {neuron_config[nt + '_W2'][0]}|{neuron_config[nt + '_W2'][1]}\n{nt}_W_3: {neuron_config[nt + '_W3'][0]}|{neuron_config[nt + '_W3'][1]}"
                annotation_string = f"{nt}_gain: {neuron_config[nt + '_GAIN'][0]}|{neuron_config[nt + '_GAIN'][1]}\n{nt}_leak: {neuron_config[nt + '_LEAK'][0]}|{neuron_config[nt + '_LEAK'][1]}"
                annotation_string += f"\n{nt}_ref: {neuron_config[nt + '_REF'][0]}|{neuron_config[nt + '_REF'][1]}\n{nt}_spk_thr: {neuron_config[nt + '_SPK_THR'][0]}|{neuron_config[nt + '_SPK_THR'][1]}"
                annotation_string += f"\n{nt}_ampa_tau: {neuron_config[nt + '_AMPA_TAU'][0]}|{neuron_config[nt + '_AMPA_TAU'][1]}\n{nt}_ampa_gain: {neuron_config[nt + '_AMPA_GAIN'][0]}|{neuron_config[nt + '_AMPA_GAIN'][1]}"
                annotation_string += f"\n{nt}_W_0: {neuron_config[nt + '_W0'][0]}|{neuron_config[nt + '_W0'][1]}\n{nt}_W_1: {neuron_config[nt + '_W1'][0]}|{neuron_config[nt + '_W1'][1]}"
                annotation_string += f"\n{nt}_DC: {neuron_config[nt + '_DC']}"
                if nt == 'PC':
                    annotation_string_2 = f"{nt}_adaptation: {neuron_config[nt + '_Adaptation']}"
                    annotation_string_2 += f"\n{nt}_pwtau: {neuron_config[nt + '_SOAD_PWTAU_N'][0]}|{neuron_config[nt + '_SOAD_PWTAU_N'][1]}"
                    annotation_string_2 += f"\n{nt}_gain: {neuron_config[nt + '_SOAD_GAIN_P'][0]}|{neuron_config[nt + '_SOAD_GAIN_P'][1]}"
                    annotation_string_2 += f"\n{nt}_tau: {neuron_config[nt + '_SOAD_TAU_P'][0]}|{neuron_config[nt + '_SOAD_TAU_P'][1]}"
                    annotation_string_2 += f"\n{nt}_w: {neuron_config[nt + '_SOAD_W_N'][0]}|{neuron_config[nt + '_SOAD_W_N'][1]}"
                    annotation_string_2 += f"\n{nt}_casc: {neuron_config[nt + '_SOAD_CASC_P'][0]}|{neuron_config[nt + '_SOAD_CASC_P'][1]}"
                    plt.text(.12*3+.01,-.09,annotation_string_2, transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox=annotation_box_config)
                if nt == 'PV': 
                    annotation_string_3 = f"{nt}_STD: {neuron_config['STD']}"
                    annotation_string_3 += f"\n{nt}_SYAM_STDW_N: {neuron_config['SYAM_STDW_N'][0]}|{neuron_config['SYAM_STDW_N'][1]}"
                    annotation_string_3 += f"\n{nt}_SYAW_STDSTR_N: {neuron_config['SYAW_STDSTR_N'][0]}|{neuron_config['SYAW_STDSTR_N'][1]}"
                    plt.text(.12*3+.01,-.15,annotation_string_3, transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox=annotation_box_config)
                plt.text(.12*i, -.09, annotation_string, transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox=annotation_box_config)



def annotate_plot_network(neuron_types,neuron_counts,neuron_config,ax):
    down_coordinate=.15
    annotation_box_config = dict(boxstyle="round", fc="w")
    annotation_string_con=f"Connections:"
    annotation_string_con += f"\nIn_PC:  {neuron_config['Input_PC']}"
    annotation_string_con += f"\nIn_PV:  {neuron_config['Input_PV']}"
    annotation_string_con += f"\nIn_SST:  {neuron_config['Input_SST']}"
    annotation_string_con += f"\nPC_PC:  {neuron_config['PC_PC']}"
    annotation_string_con += f"\nPC_PV:  {neuron_config['PC_PV']}"
    annotation_string_con += f"\nPC_SST:  {neuron_config['PC_SST']}"
    annotation_string_con += f"\nPV_PV:  {neuron_config['PV_PV']}"
    annotation_string_con += f"\nPV_PC:  {neuron_config['PV_PC']}"
    annotation_string_con += f"\nPV_SST:  {neuron_config['PV_SST']}"
    annotation_string_con += f"\nSST_PC:  {neuron_config['SST_PC']}"
    annotation_string_con += f"\nSST_PV:  {neuron_config['SST_PV']}"
    annotation_string_con += f"\n\nNVN:  {neuron_config['nvn']}"
    plt.text(0.0, -down_coordinate, annotation_string_con, size=11, transform=ax.transAxes, va="top", ha="left", bbox=annotation_box_config)
    for i, nt in enumerate(neuron_types):
            if neuron_counts[nt] > 0:
                annotation_string = f"{nt}_W_0: {neuron_config[nt + '_W0'][0]}|{neuron_config[nt + '_W0'][1]}\n{nt}_W_1: {neuron_config[nt + '_W1'][0]}|{neuron_config[nt + '_W1'][1]}"
                annotation_string += f"\n{nt}_W_2: {neuron_config[nt + '_W2'][0]}|{neuron_config[nt + '_W2'][1]}\n{nt}_W_3: {neuron_config[nt + '_W3'][0]}|{neuron_config[nt + '_W3'][1]}"

                annotation_string += f"\n{nt}_gain: {neuron_config[nt + '_GAIN'][0]}|{neuron_config[nt + '_GAIN'][1]}\n{nt}_leak: {neuron_config[nt + '_LEAK'][0]}|{neuron_config[nt + '_LEAK'][1]}"
                annotation_string += f"\n{nt}_ref: {neuron_config[nt + '_REF'][0]}|{neuron_config[nt + '_REF'][1]}\n{nt}_spk_thr: {neuron_config[nt + '_SPK_THR'][0]}|{neuron_config[nt + '_SPK_THR'][1]}"
                
                annotation_string += f"\n{nt}_ampa_tau: {neuron_config[nt + '_AMPA_TAU'][0]}|{neuron_config[nt + '_AMPA_TAU'][1]}\n{nt}_ampa_gain: {neuron_config[nt + '_AMPA_GAIN'][0]}|{neuron_config[nt + '_AMPA_GAIN'][1]}"
                annotation_string += f"\n{nt}_gaba_tau: {neuron_config[nt + '_GABA_TAU'][0]}|{neuron_config[nt + '_GABA_TAU'][1]}\n{nt}_gaba_gain: {neuron_config[nt + '_GABA_GAIN'][0]}|{neuron_config[nt + '_GABA_GAIN'][1]}"
                annotation_string += f"\n{nt}_shunt_tau: {neuron_config[nt + '_SHUNT_TAU'][0]}|{neuron_config[nt + '_SHUNT_TAU'][1]}\n{nt}_shunt_gain: {neuron_config[nt + '_SHUNT_GAIN'][0]}|{neuron_config[nt + '_SHUNT_GAIN'][1]}"
                if nt == 'PC':
                    annotation_string += f"\n{nt}_nmda_tau: {neuron_config[nt + '_NMDA_TAU'][0]}|{neuron_config[nt + '_NMDA_TAU'][1]}\n{nt}_nmda_gain: {neuron_config[nt + '_NMDA_GAIN'][0]}|{neuron_config[nt + '_NMDA_GAIN'][1]}"
                    annotation_string_2 = f"{nt}_adaptation: {neuron_config[nt + '_Adaptation']}"
                    annotation_string_2 += f"\n{nt}_pwtau: {neuron_config[nt + '_SOAD_PWTAU_N'][0]}|{neuron_config[nt + '_SOAD_PWTAU_N'][1]}"
                    annotation_string_2 += f"\n{nt}_gain: {neuron_config[nt + '_SOAD_GAIN_P'][0]}|{neuron_config[nt + '_SOAD_GAIN_P'][1]}"
                    annotation_string_2 += f"\n{nt}_tau: {neuron_config[nt + '_SOAD_TAU_P'][0]}|{neuron_config[nt + '_SOAD_TAU_P'][1]}"
                    annotation_string_2 += f"\n{nt}_w: {neuron_config[nt + '_SOAD_W_N'][0]}|{neuron_config[nt + '_SOAD_W_N'][1]}"
                    annotation_string_2 += f"\n{nt}_casc: {neuron_config[nt + '_SOAD_CASC_P'][0]}|{neuron_config[nt + '_SOAD_CASC_P'][1]}"
                    plt.text(.15*4,-down_coordinate,annotation_string_2, transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=annotation_box_config)
                if nt == 'PV': 
                    annotation_string_3 = f"{nt}_STD: {neuron_config['STD']}"
                    annotation_string_3 += f"\n{nt}_SYAM_STDW_N: {neuron_config['SYAM_STDW_N'][0]}|{neuron_config['SYAM_STDW_N'][1]}"
                    annotation_string_3 += f"\n{nt}_SYAW_STDSTR_N: {neuron_config['SYAW_STDSTR_N'][0]}|{neuron_config['SYAW_STDSTR_N'][1]}"
                    plt.text(.15*5,-down_coordinate,annotation_string_3, transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=annotation_box_config)
                plt.text(.15*i+.12, -down_coordinate, annotation_string, transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=annotation_box_config)

         
    