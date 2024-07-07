import nest
import matplotlib.pyplot as plt
from datetime import datetime

# Reset NEST environment
nest.ResetKernel()

# Create LIF neurons
num_neurons = 1024
neurons = nest.Create("iaf_psc_alpha", num_neurons)
#neurons[0].set(I_e=376.0)  # This neuron will spike due to external current
neurons[0:512].set(I_e=376.0) 

# source_neuron = neurons[0]
# connections = nest.GetConnections(source_nodes=[source_neuron])


# Define synapse specifications
syn_spec = {
    "weight": 1.0,  # Increased weight to ensure post-synaptic potential can trigger firing
    "delay": 1.5,
    
}

conn_spec = {
    "rule": "pairwise_bernoulli",
    "p": 0.1,  # 10% probability of connection between any pair of neurons
    "allow_autapses": False,
}

# Connect the first neuron to the second neuron directly
nest.Connect(neurons, neurons, conn_spec, syn_spec)

# Create a multimeter to record the membrane potentials
multimeter = nest.Create("multimeter", params={"record_from": ["V_m"]})
nest.Connect(multimeter, neurons)

# Create a spike detector
spikedetector = nest.Create("spike_recorder")
nest.Connect(neurons, spikedetector)

# Simulate for 1000 ms
nest.Simulate(1000)

# Retrieve and plot data from the multimeter
data = nest.GetStatus(multimeter)[0]["events"]
V_m = data["V_m"]
times = data["times"]
senders = data["senders"]

# Plot of membrane potential over time for all neurons
plt.figure(figsize=(12, 8))
for neuron_id in neurons.global_id:
    mask = senders == neuron_id
    plt.plot(times[mask], V_m[mask], label=f'Membrane Potential of Neuron {neuron_id}')

plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Membrane Potential Over Time')
plt.legend()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
membrane_potential_filename = f"tempsave/{timestamp}_all_neurons_membrane_potential.png"
plt.savefig(membrane_potential_filename)

# Retrieve and plot data from the spike detector
spike_data = nest.GetStatus(spikedetector)[0]["events"]
spike_times = spike_data["times"]
spike_senders = spike_data["senders"]

plt.figure(figsize=(12, 8))
plt.plot(spike_times, spike_senders, 'o')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')
plt.title('Spikes Over Time')
spike_filename = f"tempsave/{timestamp}_spikes.png"
plt.savefig(spike_filename)
