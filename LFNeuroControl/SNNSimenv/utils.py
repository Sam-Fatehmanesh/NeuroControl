import nest
import gymnasium
from gymnasium import spaces
import numpy as np
import nest
import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt


def rand_connect_neurons(source, target, weight_factor, connection_probability, delay):
    weight = nest.random.normal(mean=1.0, std=0.25) * weight_factor
    nest.Connect(source, target, {"rule": "pairwise_bernoulli", "p": connection_probability, "allow_autapses": False}, {"weight": weight, "delay": delay})

def external_rand_input_stimulation(neurons, fraction_stimulated, stimulation_probability, step_action_observsation_simulation_time, stimulator_synapse_weight, synapse_delay):
    num_stimulated = int(len(neurons) * fraction_stimulated)
    stimulated_neurons = np.random.choice(neurons, num_stimulated, replace=False)
    stimulation_times = {
        neuron: np.array(np.where(np.random.uniform(0, 1, size=int(step_action_observsation_simulation_time)) < stimulation_probability)[0] + 1, dtype=float)
        for neuron in stimulated_neurons
    }

    for neuron_id, times in stimulation_times.items():
        if len(times) > 0:
            stim_gen = nest.Create("spike_generator", params={"spike_times": times})
            nest.Connect(stim_gen, [neuron_id], syn_spec={"delay": synapse_delay, "weight": stimulator_synapse_weight})

def calculate_spike_rates(spike_data):
    senders = spike_data['senders']
    times = spike_data['times']
    total_time = times.max() - times.min()  # Total observation time
    
    # Find unique neurons and initialize spike count
    unique_neurons = np.unique(senders)
    spike_counts = {neuron: 0 for neuron in unique_neurons}
    
    # Count spikes for each neuron
    for sender in senders:
        spike_counts[sender] += 1
        
    # Multiplied by a thousand since the rates are per milisecond
    # Calculate average spike rate for each neuron
    average_rates = {neuron: 1000 * spike_counts[neuron] / total_time for neuron in unique_neurons}
    return average_rates

def graph_spikes(spike_data):
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
    plt.savefig("")