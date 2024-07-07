import nest
import gymnasium
from gymnasium import spaces
import numpy as np
import nest
import os
import json
import pickle
from datetime import datetime

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
    
    # Calculate average spike rate for each neuron
    average_rates = {neuron: spike_counts[neuron] / total_time for neuron in unique_neurons}
    # Multiplied by a thousand since the rates are per milisecond
    return average_rates * 1000

class snnEnv(gymnasium.Env):   
    def __init__(self, snn_params, neuron_params, rl_params, snn_filename=None):
        super(snnEnv, self).__init__()

        self.observation_space = spaces.MultiBinary(self.num_recorded_neurons * self.step_action_observsation_simulation_time)
        self.action_space = spaces.MultiBinary(self.num_neurons_stimulated * self.step_action_observsation_simulation_time)

        self.snn_filename = snn_filename
        self.newSNN = False
        self.neuron_params = neuron_params
        self.snn_params = snn_params
        self.current_time_step = 0
        self.current_step = 0
        self.steps_per_ep = rl_params["steps_per_ep"]
        
        # Parameters for the network
        self.num_neurons = snn_params["num_neurons"] # Number of neurons in SNN simulation
        self.inhibitory_exist = snn_params["inhibitory_exist"] # Determins if inhibitory neurons will actually be inhibitory or not.
        self.fraction_inhibitory = snn_params["fraction_inhibitory"]  # Fraction of inhibitory neurons
        self.step_action_observsation_simulation_time = snn_params["step_action_observsation_simulation_time"]  # Total length of simulation
        self.noise_rate = snn_params["noise_rate"] # Rate of noise stimulation in hertz
        self.neuron_connection_probability = snn_params["neuron_connection_probability"]  # probability that there exists a connection directed from any one neuron to any other neuron
        self.synapse_delay_time_length = snn_params["synapse_delay_time_length"] # time delay for spike signals for all synapses on all neurons
        self.synapse_weight_factor = snn_params["synapse_weight_factor"] # 
        self.noise_weight = snn_params["noise_weight"]
        self.fraction_stimulated = snn_params["fraction_stimulated"]  # Fraction of neurons to receive extra stimulation
        self.stimulation_probability = snn_params["stimulation_probability"]  # Chance of stimulation at any time step
        self.stimulator_synapse_weight = snn_params["stimulator_synapse_weight"]
        self.stimulation_time_resolution = snn_params["stimulation_time_resolution"] # 
        self.num_recorded_neurons = snn_params["num_recorded_neurons"]

        self.num_neuron_inhibitory = int(self.num_neurons * self.fraction_inhibitory)
        self.num_neuron_excitatory = self.num_neurons - self.num_inhibitory
        self.excitatory_neurons = nest.Create("iaf_psc_exp", self.num_neuron_excitatory, params=neuron_params)
        self.inhibitory_neurons = nest.Create("iaf_psc_exp", self.num_neuron_excitatory, params=neuron_params)
        self.neurons = self.inhibitory_neurons + self.excitatory_neurons

        self.neuron_2d_pos = None
        
        self.num_neurons_stimulated = int(self.num_neurons * self.fraction_stimulated)

        self.reset()


    def score(self, true_spikes, target=None):
        avg_rates = calculate_spike_rates(true_spikes)
        number_scored_neurons = 10
        assert number_scored_neurons <= self.num_recorded_neurons, "number of scored neurons can not be greater than number of recorded neurons"
        desired_rates = 200 * np.ones(number_scored_neurons) #hertz
        score_factor = 0.1
        score = np.sum((score_factor * (avg_rates[0:number_scored_neurons]-desired_rates))**2)
        return score

    def step(self, action):
        self.current_step += 1
        self.current_time_step = self.current_step * self.step_action_observsation_simulation_time
        stimulation_times = np.array(np.where(action > 0)[0] + 1, dtype=float)
        for i, neuron_id in enumerate(self.neurons):
            times = stimulation_times[i] + self.current_time_step
            if len(times) > 0:
                stim_gen = nest.Create("spike_generator", params={"spike_times": times})
                nest.Connect(stim_gen, [neuron_id], syn_spec={"delay": synapse_delay, "weight": stimulator_synapse_weight})
        
        nest.Simulate(self.step_action_observsation_simulation_time)

        spikes = nest.GetStatus(spike_recorder, "events")[0]
        reward = self.score(spikes)
        

        




        terminated = self.current_step >= self.steps_per_ep
        return observation, reward, terminated, False, None

    def reset(self):
        self.current_time_step = 0
        self.current_step = 0

        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": snn_params["stimulation_time_resolution"], "print_time": True})
        
        if self.snn_filename==None:
            self.newSNN = True

            # Randomly Connect excitatory and inhibitory neurons
            rand_connect_neurons(self.excitatory_neurons, self.excitatory_neurons, self.synapse_weight_factor)
            rand_connect_neurons(self.excitatory_neurons, self.inhibitory_neurons, self.synapse_weight_factor)

            if inhibitory_exist:
                rand_connect_neurons(self.inhibitory_neurons, self.inhibitory_neurons, -self.synapse_weight_factor)
                rand_connect_neurons(self.inhibitory_neurons, self.excitatory_neurons, -self.synapse_weight_factor)
            else:
                rand_connect_neurons(self.inhibitory_neurons, self.inhibitory_neurons, self.synapse_weight_factor)
                rand_connect_neurons(self.inhibitory_neurons, self.excitatory_neurons, self.synapse_weight_factor)

            self.neuron_2d_pos = np.random.uniform(size=(self.num_neurons, 2))

        else:
            # Load network topology from a pickle file
            with open(snn_filename, 'rb') as file:
                network_data = pickle.load(file)
            
            # Apply loaded synaptic weights and connections
            for conn_data in network_data['connections']:
                pre_neuron = conn_data['source']
                post_neuron = conn_data['target']
                weight = conn_data['weight']
                delay = conn_data['delay']

                # Establish connections with the loaded properties
                nest.Connect([pre_neuron], [post_neuron], syn_spec={'weight': weight, 'delay': delay})
            
            # You may also want to load and set neuron properties if they were saved
            for neuron_id, props in network_data['neurons'].items():
                nest.SetStatus([neuron_id], props)
        
        self.noise = nest.Create("poisson_generator", params={"rate": self.noise_rate})
        nest.Connect(self.noise, self.excitatory_neurons + self.inhibitory_neurons, syn_spec={"delay": self.synapse_delay_time_length, "weight": noise_weight})

        # Create multimeter and spike recorder
        self.multimeter = nest.Create("multimeter", params={"record_from": ["V_m"]})
        nest.Connect(self.multimeter, self.excitatory_neurons + self.inhibitory_neurons)
        self.spike_recorder = nest.Create("spike_recorder")
        nest.Connect(self.excitatory_neurons + self.inhibitory_neurons, self.spike_recorder)

        return (None, None)

    def close(self):
        if self.snn_filename is None:
            # Creating a dictionary to hold the network data
            network_data = {
                'connections': [],
                'neurons': {}
            }

            # Extracting connection data
            for conn in nest.GetConnections():
                conn_info = nest.GetStatus(conn)
                for info in conn_info:
                    network_data['connections'].append({
                        'source': info['source'],
                        'target': info['target'],
                        'weight': info['weight'],
                        'delay': info['delay']
                    })

            # Extracting neuron properties
            for neuron in self.neurons:
                neuron_props = nest.GetStatus([neuron])
                network_data['neurons'][neuron] = neuron_props[0]

            # Save the network data to a pickle file
            saved_filename = str(datetime.now())+"_snn_network_data.pkl"
            with open(saved_filename, 'wb') as file:
                pickle.dump(network_data, file)

            #print("Network saved to snn_network_data.pkl.")

            

                
