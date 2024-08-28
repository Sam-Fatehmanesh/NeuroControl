import nest
import gymnasium
from gymnasium import spaces
import numpy as np
import nest
import os
import json
import pickle
from datetime import datetime
from NeuroControl.SNNSimenv.synthCI import create_synth_frames, group_spikes_camera_fps_adjust
from NeuroControl.SNNSimenv.utils import *
import pdb
import cv2
from datetime import datetime



class snnEnv(gymnasium.Env):
    def __init__(self, snn_params, neuron_params, rl_params, apply_optical_error = True, snn_filename=None):
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
        self.reward_record = []
        self.reward_buffer = []
        self.ema_score_range = 0
        self.alpha = 0.1  # This should be 0.01 to match the 0.99 in the formula (1 - 0.99 = 0.01)
        self.minimum_rewards = 20  # Increased to ensure stable percentile calculations
        self.buffer_max_size = 4096

        self.buffer_min = 1
    

        self.apply_optical_error = apply_optical_error

        # Reset and configure the NEST kernel
        #nest.local_num_threads = 16
        # nest.ResetKernel()
        # nest.SetKernelStatus({
        #     "resolution": self.snn_params["stimulation_time_resolution"], 
        #     "print_time": True,
        #     "min_delay": 0.1,  # Adjust as necessary
        #     "max_delay": 20.0,  # Adjust as necessary
        # })

        # Parameters for the network
        self.num_neurons = snn_params["num_neurons"]  # Number of neurons in SNN simulation
        self.inhibitory_exist = snn_params["inhibitory_exist"]  # Whether inhibitory neurons are actually inhibitory
        self.fraction_inhibitory = snn_params["fraction_inhibitory"]  # Fraction of inhibitory neurons
        self.step_action_observsation_simulation_time = snn_params["step_action_observsation_simulation_time"]  # Simulation time per step
        self.noise_rate = snn_params["noise_rate"]  # Rate of noise stimulation in Hz
        self.neuron_connection_probability = snn_params["neuron_connection_probability"]  # Connection probability between neurons
        self.synapse_delay_time_length = snn_params["synapse_delay_time_length"]  # Synapse delay time
        self.synapse_weight_factor = snn_params["synapse_weight_factor"]  # Synapse weight factor
        self.ih_synapse_weight_factor = snn_params["ih_synapse_weight_factor"]
        self.noise_weight = snn_params["noise_weight"]  # Weight of noise stimulation
        self.fraction_stimulated = snn_params["fraction_stimulated"]  # Fraction of neurons to receive extra stimulation
        self.stimulation_probability = snn_params["stimulation_probability"]  # Probability of stimulation at any time step
        self.stimulator_synapse_weight = snn_params["stimulator_synapse_weight"]  # Synapse weight for stimulators
        self.stimulation_time_resolution = snn_params["stimulation_time_resolution"]  # Time resolution for stimulation
        self.num_recorded_neurons = snn_params["num_recorded_neurons"]  # Number of recorded neurons
        self.auto_ih = snn_params["auto_ih"]

        # Calculate number of excitatory and inhibitory neurons
        self.num_neuron_inhibitory = int(self.num_neurons * self.fraction_inhibitory)
        self.num_neuron_excitatory = self.num_neurons - self.num_neuron_inhibitory

        # Create neurons
        self.excitatory_neurons = nest.Create("iaf_psc_exp", self.num_neuron_excitatory, params=neuron_params)
        self.inhibitory_neurons = nest.Create("iaf_psc_exp", self.num_neuron_inhibitory, params=neuron_params)
        self.neurons = self.inhibitory_neurons + self.excitatory_neurons

        self.camera_fps = 1000#280  # Frames per second for the camera
        self.image_m, self.image_n = 480, 480  # Image dimensions

        self.neuron_2d_pos = None  # 2D positions of neurons
        self.num_neurons_stimulated = int(self.num_neurons * self.fraction_stimulated)  # Number of neurons to be stimulated
        self.num_stimulators = self.num_neurons_stimulated  # Number of stimulators
        self.stimulators = []

        # Define observation and action spaces
        frame_camera_factor = min(self.camera_fps / 1000, 1.0) // int(1000/self.camera_fps)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(int(int(self.step_action_observsation_simulation_time * frame_camera_factor)), self.image_m, self.image_n))
        self.action_space = spaces.MultiBinary(self.num_neurons_stimulated * self.step_action_observsation_simulation_time)

        # Generate random 2D positions for neurons
        self.neuron_2d_pos = np.random.rand(self.num_neurons, 2)
        self.first_reset = True
        self.firstobs, _ = self.reset()

    def score(self, true_spikes, target=None):
        """
        Calculate the score based on spike rates.
        
        Parameters:
            true_spikes (dict): Dictionary of spike events with neuron ids and times.
            target (None): Placeholder for target spike rates (not used).
        
        Returns:
            float: Calculated score based on the difference from desired spike rates.
        """
        # Number of neurons to score
        number_scored_neurons = 10
        avg_rates = calculate_spike_rates(true_spikes, number_scored_neurons)
        #pdb.set_trace()
        assert number_scored_neurons <= self.num_recorded_neurons, "number of scored neurons cannot be greater than number of recorded neurons"
        desired_rates = 200 * np.ones(number_scored_neurons)  # Desired spike rates in Hz
        #pdb.set_trace()

        # Calculate the score as the sum of squared differences from desired rates
        # avg_rates = list(avg_rates.values())[-number_scored_neurons:]
        # if len(avg_rates) < len(desired_rates):
        #     # Multiplying by -100 to make sure that no matter the desired rate there is punishment for neurons that are not firing at all and thus have no spike recoordings d
        #     avg_rates.append(-100 * np.ones(len(desired_rates) - len(avg_rates)))
        
        score = np.mean(list(avg_rates.values())[-number_scored_neurons:])#-np.sum(np.abs(self.score_factor * (list(avg_rates.values())[-number_scored_neurons:] - desired_rates)))
        #score = np.exp(-score * 0.01)
        return score

    def calculate_normalized_score(self, reward):
        self.reward_record.append(reward)

        # Checks if self.reward_buffer has more than 1000 elements, if so, the oldest element, aka the first, is removed
        if len(self.reward_buffer) < self.buffer_max_size:
            #self.reward_buffer = self.reward_buffer[1:]
            self.reward_buffer.append(reward)

            ma = np.max(self.reward_buffer)
            self.buffer_min = np.min(self.reward_buffer)

            # Calculate the range
            current_range = ma - self.buffer_min

            if current_range == 0:
                current_range = 1
        
            # Update EMA of the range
            self.ema_score_range = self.alpha * current_range + (1 - self.alpha) * self.ema_score_range
        
        # Normalize the reward using the EMA range
        normalized_score = (reward - self.buffer_min) / self.ema_score_range

    
        return normalized_score

    def GenFramesFromSpikes(self, spikes, total_sim_steps = None):
        """
        Generate observation frames from spike data.
        
        Parameters:
            spikes (dict): Dictionary of spike events with neuron ids and times.
        
        Returns:
            list: List of observation frames normalized between 0 and 1.
        """
        if total_sim_steps == None:
            total_sim_steps = self.step_action_observsation_simulation_time

        # Adjust spikes for camera FPS and generate frames
        camera_sim_adjusted_spikes = group_spikes_camera_fps_adjust(spikes, self.camera_fps)
        #pdb.set_trace()
        observation = create_synth_frames(camera_sim_adjusted_spikes, self.neuron_2d_pos, total_sim_steps, apply_distortions=self.apply_optical_error)
        #observation = [np.clip(o, 0, 255).astype(np.uint8) for o in observation]
        if self.camera_fps > 999:
            return observation
        ms_per_frame = int(1000/self.camera_fps)
        return observation[::ms_per_frame]

    def simRun(self, spikeinputs):
        """
        Runs snn simulation for time defined by self.step_action_observsation_simulation_time
        
        Parameters:
            spikeinputs (np.ndarray): neuron spike input array indicating which neurons to stimulate.
        
        Returns:
            dictionary: spikes, measured from snn activity.
        """

        self.current_step += 1
        self.current_time_step = self.current_step * self.step_action_observsation_simulation_time

        if spikeinputs.any():
            for i, sg in enumerate(self.stimulators):
                #pdb.set_trace()
                stimulation_times = np.array(np.where(spikeinputs[i] > 0)[0] + 1, dtype=float)
                times = stimulation_times + self.current_time_step
                if len(times) > 0:
                    nest.SetStatus(sg, {"spike_times": times})
        # Simulate the network
        nest.Simulate(self.step_action_observsation_simulation_time)

        spikes = self.getObsSpikes(self.step_action_observsation_simulation_time)



        return spikes

    def getObsSpikes(self, last_sim_steps):
        # Get spike events
        spikes = nest.GetStatus(self.spike_recorder, "events")[0]

        # Get the last set of spike events within the time window
        times = np.array(spikes['times'])
        #max_time = spikes['times'].max()
        # Get the maximum time from the spike events
        max_time = np.max(times)
        #indices = np.where(spikes['times'] >= (max_time - last_sim_steps))
        # Get the indices of spike events that occurred within the last simulation step
        indices = np.where(times >= (max_time - last_sim_steps))

        times = times[indices]

        # The minimum is subtracted to make frame index within observation length range
        #spikes['times'][indices] -= np.min(spikes['times'][indices])
        # Subtract the minimum time from the spike times to make them start from 0
        times -= np.min(times)

        spikes = {'senders': spikes['senders'][indices], 'times': times}

        return spikes

    def step(self, action):
        """
        Perform one step in the environment.
        
        Parameters:
            action (np.ndarray): Action array indicating which neurons to stimulate.
        
        Returns:
            tuple: Observation, reward, terminated flag, truncated flag, and additional info.
        """

        spikes = self.simRun(action)

        # Generate observation frames
        observation = self.GenFramesFromSpikes(spikes)

        # Calculate reward
        reward = self.score(spikes)
        reward = self.calculate_normalized_score(reward)

        terminated = self.current_step >= self.steps_per_ep
        
        #!!!!!!!!!!!!!!!! temp thing rn
        terminated = False

        #Since 

        return observation, reward, terminated, False

    def reset(self):
        """
        Reset the environment to an initial state.
        
        Returns:
            np.ndarray: Initial observation.
        """

        # self.current_time_step = 0
        # self.current_step = 0

        if self.first_reset:
            

            # Create noise and recording devices
            self.noise = nest.Create("poisson_generator", params={"rate": self.noise_rate})
            self.multimeter = nest.Create("multimeter", params={"record_from": ["V_m"]})
            self.spike_recorder = nest.Create("spike_recorder")

            if self.snn_filename is None:
                self.first_reset = False
                # Randomly connect neurons if no file is provided
                rand_connect_neurons(self.excitatory_neurons, self.excitatory_neurons, self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)
                rand_connect_neurons(self.excitatory_neurons, self.inhibitory_neurons, self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)

                if self.inhibitory_exist:
                    if self.auto_ih:
                        rand_connect_neurons(self.inhibitory_neurons, self.inhibitory_neurons, -self.synapse_weight_factor * self.ih_synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)
                    rand_connect_neurons(self.inhibitory_neurons, self.excitatory_neurons, -self.synapse_weight_factor * self.ih_synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)
                else:
                    rand_connect_neurons(self.inhibitory_neurons, self.inhibitory_neurons, self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)
                    rand_connect_neurons(self.inhibitory_neurons, self.excitatory_neurons, self.synapse_weight_factor, self.neuron_connection_probability, self.synapse_delay_time_length)



                # Connect noise generator to neurons
                nest.Connect(self.noise, self.excitatory_neurons + self.inhibitory_neurons, syn_spec={"delay": self.synapse_delay_time_length, "weight": self.noise_weight})

                # Connect recording devices to neurons
                nest.Connect(self.multimeter, self.excitatory_neurons + self.inhibitory_neurons)
                nest.Connect(self.excitatory_neurons + self.inhibitory_neurons, self.spike_recorder)

                

            else :
                self.first_reset = False
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

            for i, neuron_id in list(enumerate(self.neurons))[:self.num_neurons_stimulated]:
                #stimulation_times = np.array(np.where(spikeinputs[i] > 0)[0] + 1, dtype=float)
                #times = stimulation_times + self.current_time_step
                stim_gen = nest.Create("spike_generator")
                nest.Connect(stim_gen, neuron_id, syn_spec={"delay": self.synapse_delay_time_length, "weight": self.stimulator_synapse_weight})
                self.stimulators.append(stim_gen)

        # Simulate initial state
        nest.Simulate(self.step_action_observsation_simulation_time)
        spikes = nest.GetStatus(self.spike_recorder, "events")[0]

        # Generate initial observation
        observation = self.GenFramesFromSpikes(spikes)

        # Reset spike recorder
        self.cleanSpikeRecorder()

        return observation, None

    def close(self, dirprefix = None):
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
            if dirprefix is None:
                dirprefix = str(datetime.now()) +"_"

            saved_filename = dirprefix + "snn_network_data.pkl"
            with open(saved_filename, 'wb') as file:
                pickle.dump(network_data, file)

    def seed(self, seed):
        np.random.seed(seed)

    # Generates a video render of the simulation and generates a graphic of the spikes of the different neurons over time
    def render(self, past_sim_steps_n, save_dir="renders/", fps=1.0):
        # Check if a folder called renders exists if not create it
        if save_dir == "renders/":
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            

        spikes = self.getObsSpikes(past_sim_steps_n)
        # pdb.set_trace()
        # Generate frames from spikes
        frames = self.GenFramesFromSpikes(spikes, total_sim_steps=past_sim_steps_n)
        # convert numpy of frames into a list of frames
        frames = [frame for frame in frames]
        # print("######")
        # print(len(frames))
        # print(past_sim_steps_n)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (280, 280)  # Assuming the frames are square with edge length `image_n`
        filename = save_dir + "true_simulation.mp4"
        out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

        # Write each frame to the video
        for frame in frames:
            # Convert the frame from floating-point to 8-bit unsigned integer
            frame = frame.squeeze()  # Remove singleton dimensions if any
            frame = np.clip(frame, 0, 255).astype(np.uint8)

            if frame.ndim == 2:
                # If the frame is grayscale, convert it to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Write the frame to the video
            out.write(frame)

        # Release the VideoWriter object
        out.release()

        #Graphs and saves spikes
        # Extract spike times for each neuron
        graph_spikes(spikes, save_dir + "spikes.jpeg")


    def cleanSpikeRecorder(self):
        # Clean spike recorder
        nest.SetStatus(self.spike_recorder, {"n_events": 0})