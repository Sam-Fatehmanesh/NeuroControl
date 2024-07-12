import numpy as np
from LFNeuroControl.SNNSimenv.snnenv import snnEnv
from LFNeuroControl.SNNSimenv.synthCI import create_video
from datetime import datetime
import time
import os
import matplotlib.pyplot as plt
import json
from scipy.fftpack import fft, fftfreq
import itertools
import seaborn as sns
from scipy.signal import correlate




# Create a directory to save plots
timestamp = time.strftime("%Y%m%d-%H%M%S")
folder_name = f"./experiments/{timestamp}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
folder_name += "/"

def compute_cross_correlation_matrix(spike_data, dt=0.001):
    """
    Computes the normalized cross-correlation matrix for multiple neurons.

    Parameters:
    spike_data (dict): Dictionary with two keys:
                       'senders' - np.array of neuron identifiers
                       'times' - np.array of spike times corresponding to each sender
    dt (float): Time step for the binary time series (default is 0.001s).

    Returns:
    np.array: Normalized cross-correlation matrix.
    """
    senders = spike_data['senders']
    times = spike_data['times']
    unique_neurons = np.unique(senders)
    
    # Define the range and resolution of the time series
    t_min = np.min(times)
    t_max = np.max(times)
    t_bins = np.arange(t_min, t_max, dt)
    
    # Create a binary time series for each neuron
    spike_trains = []
    for neuron_id in unique_neurons:
        neuron_times = times[senders == neuron_id]
        spike_train, _ = np.histogram(neuron_times, bins=t_bins)
        spike_trains.append(spike_train)
    
    # Compute the cross-correlation matrix
    n_neurons = len(unique_neurons)
    cross_corr_matrix = np.zeros((n_neurons, n_neurons))
    
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                cross_corr_matrix[i, j] = 1.0  # Self-correlation is always 1
            else:
                spike_train_i = spike_trains[i] - np.mean(spike_trains[i])
                spike_train_j = spike_trains[j] - np.mean(spike_trains[j])
                cross_corr = correlate(spike_train_i, spike_train_j, mode='full')
                normalization_factor = np.std(spike_train_i) * np.std(spike_train_j) * len(spike_train_i)
                if normalization_factor != 0:
                    cross_corr_matrix[i, j] = np.max(cross_corr) / normalization_factor
                else:
                    cross_corr_matrix[i, j] = 0
    
    return cross_corr_matrix, unique_neurons

def plot_cross_correlation_heatmap(spike_data, filename, dt=0.001):
    """
    Plots the cross-correlation heatmap for multiple neurons.

    Parameters:
    spike_data (dict): Dictionary with two keys:
                       'senders' - np.array of neuron identifiers
                       'times' - np.array of spike times corresponding to each sender
    filename (str): Name of the file to save the plot.
    dt (float): Time step for the binary time series (default is 0.001s).
    """
    cross_corr_matrix, unique_neurons = compute_cross_correlation_matrix(spike_data, dt)

    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True,
                xticklabels=unique_neurons, yticklabels=unique_neurons)
    plt.xlabel('Neuron ID')
    plt.ylabel('Neuron ID')
    plt.title('Cross-Correlation Heatmap')
    plt.savefig(filename)

def graph_average_spike_frequencies(spike_data, filename, dt=0.001):
    """
    Creates an averaged frequency spectrum plot from spike times of multiple neurons using Fourier Transform.

    Parameters:
    spike_data (dict): Dictionary with two keys:
                       'senders' - np.array of neuron identifiers
                       'times' - np.array of spike times corresponding to each sender
    filename (str): Name of the file to save the plot.
    dt (float): Time step for the binary time series (default is 0.001s).
    """
    senders = spike_data['senders']
    times = spike_data['times']
    unique_neurons = np.unique(senders)
    
    # Define the range and resolution of the time series
    t_min = np.min(times)
    t_max = np.max(times)
    t_bins = np.arange(t_min, t_max, dt)

    power_spectra = []

    for neuron_id in unique_neurons:
        neuron_times = times[senders == neuron_id]
        
        # Create a binary time series indicating spikes
        spike_train, _ = np.histogram(neuron_times, bins=t_bins)
        
        # Perform Fourier transform
        fft_vals = fft(spike_train)
        fft_freqs = fftfreq(len(spike_train), dt)
        
        # Compute power spectrum
        power_spectrum = np.abs(fft_vals)**2
        power_spectra.append(power_spectrum)
    
    # Compute the average power spectrum
    average_power_spectrum = np.mean(power_spectra, axis=0)
    
    # Plot the average power spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(fft_freqs[:len(fft_freqs)//2], average_power_spectrum[:len(average_power_spectrum)//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Average Power Spectrum of Spike Times')
    plt.grid(True)
    plt.savefig(filename)


def graph_spike_frequencies(spike_data, filename, dt=0.001):
    """
    Creates a frequency spectrum plot from spike times of multiple neurons using Fourier Transform.

    Parameters:
    spike_data (dict): Dictionary with two keys:
                       'senders' - np.array of neuron identifiers
                       'times' - np.array of spike times corresponding to each sender
    filename (str): Name of the file to save the plot.
    dt (float): Time step for the binary time series (default is 0.001s).
    """
    senders = spike_data['senders']
    times = spike_data['times']
    unique_neurons = np.unique(senders)
    
    # Define the range and resolution of the time series
    t_min = np.min(times)
    t_max = np.max(times)
    t_bins = np.arange(t_min, t_max, dt)

    # Prepare for plotting
    plt.figure(figsize=(10, 6))
    colors = itertools.cycle(plt.cm.tab10.colors)  # Cycle through a set of colors
    linestyles = itertools.cycle(['-', '--', '-.', ':'])  # Cycle through a set of line styles

    for neuron_id in unique_neurons:
        neuron_times = times[senders == neuron_id]
        
        # Create a binary time series indicating spikes
        spike_train, _ = np.histogram(neuron_times, bins=t_bins)
        
        # Perform Fourier transform
        fft_vals = fft(spike_train)
        fft_freqs = fftfreq(len(spike_train), dt)
        
        # Compute power spectrum
        power_spectrum = np.abs(fft_vals)**2
        
        # Plot the power spectrum for this neuron
        color = next(colors)
        linestyle = next(linestyles)
        plt.plot(fft_freqs[:len(fft_freqs)//2], power_spectrum[:len(power_spectrum)//2], label=f'Neuron {neuron_id}', color=color, linestyle=linestyle)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power Spectrum of Spike Times')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)

def graph_spikes(spike_data, filename):
    """
    Creates a scatter plot of spikes from multiple neurons.

    Parameters:
    spike_data (dict): Dictionary with two keys:
                       'senders' - np.array of neuron identifiers
                       'times' - np.array of spike times corresponding to each sender
    """
    senders = spike_data['senders']
    times = spike_data['times']

    plt.figure(figsize=(10, 6))
    plt.scatter(times, senders, alpha=0.6, edgecolors='none', s=10)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron ID')
    plt.title('Spike Plot of Neurons')
    plt.grid(True)
    plt.savefig(filename)



def save_dicts_to_file(dict_list, file_path):
    """
    Saves a list of dictionaries to a text file in JSON format.

    Parameters:
    dict_list (list): A list of dictionaries to be saved.
    file_path (str): The path of the file where the data will be saved.
    """
    with open(file_path, 'w') as file:
        json.dump(dict_list, file, indent=4)  # 'indent' for pretty-printing





# SNN Parameters
snn_params = {
    "num_neurons": 16,
    "inhibitory_exist": True,
    "fraction_inhibitory": 0.4,
    "step_action_observsation_simulation_time": 100.0,
    "noise_rate": 800,
    "neuron_connection_probability": 0.001,
    "synapse_delay_time_length": 1.0,
    "synapse_weight_factor": 1,
    "noise_weight": 1.0,
    "fraction_stimulated": 0.2,
    "stimulation_probability": 1,
    "stimulator_synapse_weight": 1.3,
    "stimulation_time_resolution": 0.1,
    "num_recorded_neurons": 10,
    "num_neurons_stimulated": int(0.2*1024),
    "ih_synapse_weight_factor": 1,
    "auto_ih": True,
}

# RL Parameters
rl_params = {
    "steps_per_ep": 64,  # Total number of steps per episode
    "score_factor": 0.1   # Scoring factor for rewards
}

# Neuron Parameters
neuron_params = {
    "C_m": 0.25,  # nF    membrane capacitance
    "I_e": 0.5,   # nA    bias current
    "tau_m": 20.0,  # ms    membrane time constant
    "t_ref": 2.0,  # ms    refractory period
    "tau_syn_ex": 5.0,  # ms    excitatory synapse time constant
    "tau_syn_in": 5.0,  # ms    inhibitory synapse time constant
    "V_reset": -70.0,  # mV    reset membrane potential
    "E_L": -65.0,  # mV    resting membrane potential
    "V_th": -50.0  # mV    firing threshold voltage
}

save_dicts_to_file([snn_params, rl_params, neuron_params], folder_name+'hyperparams.json')

fractions_inhibitory = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.99]
num_tests = 1
for fi in fractions_inhibitory:
    for i in range(num_tests):
        snn_params["fraction_inhibitory"] = fi
        env = snnEnv(snn_params=snn_params, 
        neuron_params=neuron_params, 
        rl_params=rl_params, 
        snn_filename=None)

        spikes = env.simRun()
        graph_spikes(spikes, folder_name + str(fi)+"_fi.jpeg")
        graph_average_spike_frequencies(spikes, folder_name + str(fi)+"_fiFREQ.jpeg")
        plot_cross_correlation_heatmap(spikes, folder_name + str(fi)+"_fiCROSSCOR.jpeg")
