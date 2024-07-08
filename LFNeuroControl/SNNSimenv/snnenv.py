import nest
import gymnasium
from gymnasium import spaces
import numpy as np
import nest
import os
import json
import pickle
from datetime import datetime
from LFNeuroControl.SNNSimenv.synthCI import create_synth_frames, group_spikes_camera_fps_adjust
from LFNeuroControl.SNNSimenv.utils import *
import pdb

class snnEnv(gymnasium.Env):
    def __init__(self, snn_params, neuron_params, rl_params, snn_filename=None):
        """
        Initialize the SNN environment.
        
        Parameters:
            snn_params (dict): Parameters for the SNN configuration.
            neuron_params (dict): Parameters for individual neurons.
            rl_params (dict): Parameters for reinforcement learning.
            snn_filename (str, optional): Filename to load or save the SNN configuration.
        """
        super(snnEnv, self).__init__()

        # Store parameters and initialize counters
        self.snn_filename = snn_filename
        self.neuron_params = neuron_params
        self.rl_params = rl_params
        self.snn_params = snn_params
        self.current_time_step = 0
        self.current_step = 0
        self.steps_per_ep = rl_params["steps_per_ep"]
        self.score_factor = rl_params["score_factor"]

        # Reset and configure the NEST kernel
        nest.ResetKernel()
        nest.SetKernelStatus({
            "resolution": self.snn_params["stimulation_time_resolution"], 
            "print_time": True,
            "min_delay": 0.1,  # Adjust as necessary
            "max_delay": 20.0,  # Adjust as necessary
        })

        # Parameters for the network
        self.num_neurons = snn_params["num_neurons"]  # Number of neurons in SNN simulation
        self.inhibitory_exist = snn_params["inhibitory_exist"]  # Whether inhibitory neurons are actually inhibitory
        self.fraction_inhibitory = snn_params["fraction_inhibitory"]  # Fraction of inhibitory neurons
        self.step_action_observsation_simulation_time = snn_params["step_action_observsation_simulation_time"]  # Simulation time per step
        self.noise_rate = snn_params["noise_rate"]  # Rate of noise stimulation in Hz
        self.neuron_connection_probability = snn_params["neuron_connection_probability"]  # Connection probability between neurons
        self.synapse_delay_time_length = snn_params["synapse_delay_time_length"]  # Synapse delay time
        self.synapse_weight_factor = snn_params["synapse_weight_factor"]  # Synapse weight factor
        self.noise_weight = snn_params["noise_weight"]  # Weight of noise stimulation
        self.fraction_stimulated = snn_params["fraction_stimulated"]  # Fraction of neurons to receive extra stimulation
        self.stimulation_probability = snn_params["stimulation_probability"]  # Probability of stimulation at any time step
        self.stimulator_synapse_weight = snn_params["stimulator_synapse_weight"]  # Synapse weight for stimulators
        self.stimulation_time_resolution = snn_params["stimulation_time_resolution"]  # Time resolution for stimulation
        self.num_recorded_neurons = snn_params["num_recorded_neurons"]  # Number of recorded neurons

        # Calculate number of excitatory and inhibitory neurons
        self.num_neuron_inhibitory = int(self.num_neurons * self.fraction_inhibitory)
        self.num_neuron_excitatory = self.num_neurons - self.num_neuron_inhibitory

        # Create neurons
        self.excitatory_neurons = nest.Create("iaf_psc_exp", self.num_neuron_excitatory, params=neuron_params)
        self.inhibitory_neurons = nest.Create("iaf_psc_exp", self.num_neuron_inhibitory, params=neuron_params)
        self.neurons = self.inhibitory_neurons + self.excitatory_neurons

        self.camera_fps = 280  # Frames per second for the camera
        self.image_m, self.image_n = 480, 480  # Image dimensions

        self.neuron_2d_pos = None  # 2D positions of neurons
        self.num_neurons_stimulated = int(self.num_neurons * self.fraction_stimulated)  # Number of neurons to be stimulated
        self.num_stimulators = self.num_neurons_stimulated  # Number of stimulators
        self.stimulators = []

        # Define observation and action spaces
        frame_camera_factor = min(self.camera_fps / 1000, 1.0)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(int(self.step_action_observsation_simulation_time * frame_camera_factor), self.image_m, self.image_n), 
            dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(self.num_neurons_stimulated * self.step_action_observsation_simulation_time)

        self.reset()

    def score(self, true_spikes, target=None):
        """
        Calculate the score based on spike rates.
        
        Parameters:
            true_spikes (dict): Dictionary of spike events with neuron ids and times.
            target (None): Placeholder for target spike rates (not used).
        
        Returns:
            float: Calculated score based on the difference from desired spike rates.
        """
        avg_rates = calculate_spike_rates(true_spikes)
        number_scored_neurons = 10  # Number of neurons to score
        assert number_scored_neurons <= self.num_recorded_neurons, "number of scored neurons cannot be greater than number of recorded neurons"
        desired_rates = 200 * np.ones(number_scored_neurons)  # Desired spike rates in Hz
        
        # Calculate the score as the sum of squared differences from desired rates
        score = np.sum((self.score_factor * (list(avg_rates.values())[-number_scored_neurons:] - desired_rates))**2)
        return score

    def GenFramesFromSpikes(self, spikes):
        """
        Generate observation frames from spike data.
        
        Parameters:
            spikes (dict): Dictionary of spike events with neuron ids and times.
        
        Returns:
            list: List of observation frames normalized between 0 and 1.
        """
        # Adjust spikes for camera FPS and generate frames
        camera_sim_adjusted_spikes = group_spikes_camera_fps_adjust(spikes, self.camera_fps)
        observation = create_synth_frames(camera_sim_adjusted_spikes, self.neuron_2d_pos, self.step_action_observsation_simulation_time)
        observation = [o / 255.0 for o in observation]  # Normalize frames
        return observation

    def step(self, action):
        """
        Perform one step in the environment.
        
        Parameters:
            action (np.ndarray): Action array indicating which neurons to stimulate.
        
        Returns:
            tuple: Observation, reward, terminated flag, truncated flag, and additional info.
        """
        self.current_step += 1
        self.current_time_step = self.current_step * self.step_action_observsation_simulation_time

        # Stimulate neurons based on the action
        for i, neuron_id in list(enumerate(self.neurons))[:self.num_neurons_stimulated]:
            stimulation_times = np.array(np.where(action[i] > 0)[0] + 1, dtype=float)
            times = stimulation_times + self.current_time_step
            if len(times) > 0:
                stim_gen = nest.Create("spike_generator", params={"spike_times": times})
                nest.Connect(stim_gen, neuron_id, syn_spec={"delay": self.synapse_delay_time_length, "weight": self.stimulator_synapse_weight})
        
        # Simulate the network
        nest.Simulate(self.step_action_observsation_simulation_time)

        # Get spike events
        spikes = nest.GetStatus(self.spike_recorder, "events")[0]

        # Get the last set of spike events within the time window
        max_time = spikes['times'].max()
        time_window = self.step_action_observsation_simulation_time
        indices = np.where(spikes['times'] >= (max_time - time_window))
        spikes = {'senders': spikes['senders'][indices], 'times': spikes['times'][indices]}

        # Generate observation frames
        observation = self.GenFramesFromSpikes(spikes)

        # Calculate reward
        reward = self.score(spikes)
        terminated = self.current_step >= self.steps_per_ep
        return observation, reward, terminated, False, None

    def reset(self):
        """
        Reset the environment to an initial state.
        
        Returns:
            np.ndarray: Initial observation.
        """
        self.current_time_step = 0
        self.current_step = 0

        # Create noise and recording devices
        self.noise = nest.Create("poisson_generator", params={"rate": self.noise_rate})
        self.multimeter = nest.Create("multimeter", params={"record_from": ["V_m"]})
        self.spike_recorder = nest.Create("spike_recorder")

        if self.snn_filename is None:
            # Randomly connect neurons if no file is provided
            rand_connect_neurons(self.excitatory_neurons, self.excitatory_neurons, self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)
            rand_connect_neurons(self.excitatory_neurons, self.inhibitory_neurons, self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)

            if self.inhibitory_exist:
                rand_connect_neurons(self.inhibitory_neurons, self.inhibitory_neurons, -self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)
                rand_connect_neurons(self.inhibitory_neurons, self.excitatory_neurons, -self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)
            else:
                rand_connect_neurons(self.inhibitory_neurons, self.inhibitory_neurons, self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)
                rand_connect_neurons(self.inhibitory_neurons, self.excitatory_neurons, self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)

            # Generate random 2D positions for neurons
            self.neuron_2d_pos = np.random.uniform(size=(self.num_neurons, 2))

            # Connect noise generator to neurons
            nest.Connect(self.noise, self.excitatory_neurons + self.inhibitory_neurons, syn_spec={"delay": self.synapse_delay_time_length, "weight": self.noise_weight})

            # Connect recording devices to neurons
            nest.Connect(self.multimeter, self.excitatory_neurons + self.inhibitory_neurons)
            nest.Connect(self.excitatory_neurons + self.inhibitory_neurons, self.spike_recorder)

        else:
            # Load network topology from a pickle file
            with open(self.snn_filename, 'rb') as file:
                network_data, self.neuron_2d_pos = pickle.load(file)
            
            # Load and set neuron properties
            for neuron_id, props in network_data['neurons'].items():
                nest.SetStatus(self.neurons[neuron_id-1], props)

            # Apply loaded synaptic weights and connections
            for conn_data in network_data['connections']:
                pre_node = conn_data['source']
                post_node = conn_data['target']
                weight = conn_data['weight']
                delay = conn_data['delay']

                # Establish connections with the loaded properties
                number_of_nodes = len(nest.GetNodes()[0])
                if pre_node <= number_of_nodes and post_node <= number_of_nodes:
                    nest.Connect([pre_node], [post_node], syn_spec={'weight': weight, 'delay': delay})
            
            # Connect recording devices to neurons
            nest.Connect(self.excitatory_neurons + self.inhibitory_neurons, self.spike_recorder)

        # Simulate initial state
        nest.Simulate(self.step_action_observsation_simulation_time)
        spikes = nest.GetStatus(self.spike_recorder, "events")[0]

        # Generate initial observation
        observation = self.GenFramesFromSpikes(spikes)

        return observation

    def close(self):
        """
        Close the environment and save the network state if necessary.
        """
        if self.snn_filename is None:
            # Create a dictionary to hold the network data
            network_data = {
                'connections': [],
                'neurons': {}
            }

            # Extract connection data
            for conn in nest.GetConnections():
                conn_info = nest.GetStatus(conn)
                for info in conn_info:
                    network_data['connections'].append({
                        'source': info['source'],
                        'target': info['target'],
                        'weight': info['weight'],
                        'delay': info['delay']
                    })

            # Extract neuron properties
            for neuron in self.neurons:
                neuron_props = nest.GetStatus(neuron)
                neuron_id = neuron_props[0]["global_id"]
                network_data['neurons'][neuron_id] = neuron_props[0]
                del network_data['neurons'][neuron_id]['t_spike']
                del network_data['neurons'][neuron_id]['thread']
                del network_data['neurons'][neuron_id]['thread_local_id']
                del network_data['neurons'][neuron_id]['vp']

            # Include neuron positions in the saved data
            network_data = (network_data, self.neuron_2d_pos)

            # Save the network data to a pickle file
            saved_filename = str(datetime.now()) + "_snn_network_data.pkl"
            with open(saved_filename, 'wb') as file:
                pickle.dump(network_data, file)
