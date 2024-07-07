import nest
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# Create a directory to save plots
timestamp = time.strftime("%Y%m%d-%H%M%S")
folder_name = f"./tempsave/{timestamp}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Parameters for the network
num_neurons = 64
inhibitory_exist = True
fraction_inhibitory = 0.5  # Fraction of inhibitory neurons
simulation_time = 100.0  # in ms
noise_rate = 800
connection_probability = 0.1  # Connection probability
delay = 1.0
synapse_weight_factor = 1
noise_weight = 1.0
fraction_stimulated = 0.2  # Fraction of neurons to receive extra stimulation
stimulation_probability = 1  # Chance of stimulation at any time step
stimulator_synapse_weight = 1.3
stimulation_time_resolution = 0.1


avg_external_current = 0.5
# Neuron parameters
neuron_params = {
    "C_m": 0.25,  # nF    membrane capacitance
    "I_e": avg_external_current,  # nA    bias current
    "tau_m": 20.0,  # ms    membrane time constant
    "t_ref": 2.0,  # ms    refractory period
    "tau_syn_ex": 5.0,  # ms    excitatory synapse time constant
    "tau_syn_in": 5.0,  # ms    inhibitory synapse time constant
    "V_reset": -70.0,  # mV    reset membrane potential
    "E_L": -65.0,  # mV    resting membrane potential
    "V_th": -50.0,  # mV    firing threshold voltage
}

# Initialize NEST
nest.ResetKernel()
nest.SetKernelStatus({"resolution": stimulation_time_resolution, "print_time": True})

# Create excitatory and inhibitory neurons
num_inhibitory = int(num_neurons * fraction_inhibitory)
num_excitatory = num_neurons - num_inhibitory
excitatory_neurons = nest.Create("iaf_psc_exp", num_excitatory, params=neuron_params)
inhibitory_neurons = nest.Create("iaf_psc_exp", num_inhibitory, params=neuron_params)

# Function to connect neurons with specified properties
def connect_neurons(source, target, weight_factor):
    weight = nest.random.normal(mean=1.0, std=0.25) * weight_factor
    nest.Connect(source, target, {"rule": "pairwise_bernoulli", "p": connection_probability, "allow_autapses": False}, {"weight": weight, "delay": 1.0})

# Connect neurons
connect_neurons(excitatory_neurons, excitatory_neurons, synapse_weight_factor)
connect_neurons(excitatory_neurons, inhibitory_neurons, synapse_weight_factor)
if inhibitory_exist:
    connect_neurons(inhibitory_neurons, inhibitory_neurons, -synapse_weight_factor)
    connect_neurons(inhibitory_neurons, excitatory_neurons, -synapse_weight_factor)
else:
    connect_neurons(inhibitory_neurons, inhibitory_neurons, synapse_weight_factor)
    connect_neurons(inhibitory_neurons, excitatory_neurons, synapse_weight_factor)
# Create and connect noise generator
noise = nest.Create("poisson_generator", params={"rate": noise_rate})
nest.Connect(noise, excitatory_neurons + inhibitory_neurons, syn_spec={"delay": delay, "weight": noise_weight})

# Extra stimulation setup
def create_extra_stimulation(neurons, fraction_stimulated, stimulation_probability, simulation_time):
    num_stimulated = int(len(neurons) * fraction_stimulated)
    stimulated_neurons = np.random.choice(neurons, num_stimulated, replace=False)
    stimulation_times = {
        neuron: np.array(np.where(np.random.uniform(0, 1, size=int(simulation_time)) < stimulation_probability)[0] + 1, dtype=float)
        for neuron in stimulated_neurons
    }

    for neuron_id, times in stimulation_times.items():
        if len(times) > 0:
            stim_gen = nest.Create("spike_generator", params={"spike_times": times})
            nest.Connect(stim_gen, [neuron_id], syn_spec={"delay": delay, "weight": stimulator_synapse_weight})

create_extra_stimulation(excitatory_neurons + inhibitory_neurons, fraction_stimulated, stimulation_probability, simulation_time)

# Create multimeter and spike recorder
multimeter = nest.Create("multimeter", params={"record_from": ["V_m"]})
nest.Connect(multimeter, excitatory_neurons + inhibitory_neurons)
spike_recorder = nest.Create("spike_recorder")
nest.Connect(excitatory_neurons + inhibitory_neurons, spike_recorder)

# Run simulation
nest.Simulate(simulation_time)

# Retrieve and plot spikes
spikes = nest.GetStatus(spike_recorder, "events")[0]
print(spikes)
times = spikes["times"]
senders = spikes["senders"]
plt.figure(figsize=(12, 8))
plt.scatter(times, senders, s=2)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.title("Spike Raster Plot")
spike_plot_filename = os.path.join(folder_name, "spikes.jpeg")
plt.savefig(spike_plot_filename)
plt.close()

# Retrieve and plot membrane potentials
dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
senders = dmm["events"]["senders"]
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(senders))))
for idx, neuron_id in enumerate(np.unique(senders)):
    mask = senders == neuron_id
    plt.plot(ts[mask], Vms[mask], label=f'Neuron {neuron_id}', color=colors[idx], linestyle='-', marker='None', linewidth=2)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Membrane Potentials Over Time")
plt.legend(title="Neuron ID")
plt.grid(True)
plt.ylim(bottom=-80)
membrane_plot_filename = os.path.join(folder_name, "membrane_potentials.jpeg")
plt.savefig(membrane_plot_filename)
plt.close()

print(f"Spike raster plot saved to {spike_plot_filename}")
print(f"Membrane potential plot saved to {membrane_plot_filename}")
print(spikes)