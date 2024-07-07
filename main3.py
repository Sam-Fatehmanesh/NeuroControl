import nest
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# Create a directory to save plots
timestamp = time.strftime("%Y%m%d-%H%M%S")
folder_name = f"./tempsave/{timestamp}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Parameters
num_neurons = 4  # total number of neurons in the network
fraction_stimulated = 0.5  # fraction of neurons to be stimulated
noise_rate = 800.0  # rate of background noise in Hz
delay = 1.0  # delay for all synapses
stimulation_probability = 0.5
simulation_time = 10000.0  # in ms
connection_probability = 1.0  # probability of connection between any two neurons

# Neuron parameters
neuron_params = {
    "tau_m": 28.53,
    "V_th": 1278.0,
    "E_L": 616.0,
    "V_m": 616.0,
    "V_reset": 355.0,
    "t_ref": 3.98,
    "tau_syn_ex": 1.8,
    "C_m": 2.36,
}

# Initialize NEST and create neurons
nest.ResetKernel()
nest.SetKernelStatus({"resolution": 0.1, "print_time": True})
neurons = nest.Create("iaf_psc_exp", num_neurons, params=neuron_params)

# Create background noise
noise = nest.Create("poisson_generator", params={"rate": noise_rate})
nest.Connect(noise, neurons, syn_spec={"delay": delay, "weight": 1.6})

# Randomly connect neurons using the pairwise Bernoulli rule
nest.Connect(neurons, neurons, {"rule": "pairwise_bernoulli", "p": connection_probability},
             syn_spec={"weight": nest.random.normal(mean=1.0, std=0.5), "delay": delay})

# Determine which neurons will be stimulated
num_stimulated = int(num_neurons * fraction_stimulated)
stimulated_neurons = random.sample(range(num_neurons), num_stimulated)
stimulation_times = {
    neuron: np.array(np.where(np.random.uniform(0, 1, size=(int(simulation_time))) < stimulation_probability)[0] + 1, dtype=float)
    for neuron in stimulated_neurons
}

# Create individual spike generators for each stimulated neuron
for neuron_id, times in stimulation_times.items():
    stim_gen = nest.Create("spike_generator", params={"spike_times": times})
    nest.Connect(stim_gen, neurons[neuron_id], syn_spec={"delay": delay, "weight": 1.3})

# Create a multimeter to record membrane potentials
multimeter = nest.Create("multimeter", params={"record_from": ["V_m"]})
nest.Connect(multimeter, neurons)

# Create a spike recorder
spike_recorder = nest.Create("spike_recorder")
nest.Connect(neurons, spike_recorder)

# Run simulation
nest.Simulate(simulation_time)

# Retrieve and plot spikes
spikes = nest.GetStatus(spike_recorder, "events")[0]
times = spikes["times"]
senders = spikes["senders"]

# Plotting spike raster plot
plt.figure(figsize=(12, 8))
plt.scatter(times, senders, s=2)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.title("Spike Raster Plot")
spike_plot_filename = os.path.join(folder_name, f"{datetime.now():%Y%m%d-%H%M%S}_spike_raster_plot.png")
plt.savefig(spike_plot_filename)
plt.close()

# Retrieve and plot membrane potentials
dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
senders = dmm["events"]["senders"]

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(senders))))  # Color map for neurons
for idx, neuron_id in enumerate(np.unique(senders)):
    mask = senders == neuron_id
    plt.plot(ts[mask], Vms[mask], label=f'Neuron {neuron_id}', color=colors[idx], linestyle='-', marker='None', linewidth=2)

plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Membrane Potentials Over Time")
plt.legend(title="Neuron ID")
plt.grid(True)
membrane_plot_filename = os.path.join(folder_name, f"{datetime.now():%Y%m%d-%H%M%S}_membrane_potentials.png")
plt.savefig(membrane_plot_filename)
plt.close()

print(f"Spike raster plot saved to {spike_plot_filename}")
print(f"Membrane potential plot saved to {membrane_plot_filename}")
