# from NeuroControl.SNNSimenv.snnenv import snnEnv
import numpy as np
import torch
import os
from datetime import datetime
import time
from tqdm import tqdm
import random
import threading
import queue

class NeuralControl:
    def __init__(self, snn_params, rl_params, device):
        self.device = device
        self.snn_params = snn_params
        self.rl_params = rl_params
        
        # Set random seed
        print("Setting random seeds.")
        seed = 42
        np.random.seed(seed)
        torch.set_default_device(device)
        torch.manual_seed(seed)

        self.action_size = int(snn_params["num_neurons_stimulated"] * snn_params["step_action_observsation_simulation_time"])
        self.action_dims = (snn_params["num_neurons_stimulated"], snn_params["step_action_observsation_simulation_time"])

        self.probabilityOfSpikeAction = 0.5

        self.data_queue = queue.Queue(maxsize=20000)
        self.stop_generation = False
        self.simulation_thread = None

    def simulation_worker(self):
        import nest
        from NeuroControl.SNNSimenv.snnenv import snnEnv

        print("Initializing NEST backend with reduced verbosity.")
        nest.ResetKernel()
        #nest.set_verbosity("M_ERROR")

        sim = snnEnv(snn_params=self.snn_params, 
                     neuron_params=self.neuron_params, 
                     rl_params=self.rl_params, 
                     snn_filename=None,
                     apply_optical_error=False)

        obs, _ = sim.reset()
        
        while not self.stop_generation:
            action = np.random.rand(*self.action_dims) > self.probabilityOfSpikeAction
            next_obs, reward, done, _ = sim.step(action)

            if self.data_queue.full():
                self.data_queue.get()  # Remove oldest item if queue is full
            self.data_queue.put((obs, action, reward, next_obs, done))

            obs = next_obs

            if self.data_queue.qsize() % 16 == 0:
                sim.cleanSpikeRecorder()

            if done:
                obs, _ = sim.reset()

    def start_data_generation(self):
        self.simulation_thread = threading.Thread(target=self.simulation_worker)
        self.simulation_thread.start()

    def stop_data_generation(self):
        self.stop_generation = True
        if self.simulation_thread:
            self.simulation_thread.join()

    def sample_buffer(self, batch_size):
        if self.data_queue.qsize() < batch_size:
            return None

        sampled_data = random.sample(list(self.data_queue.queue), batch_size)
        
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*sampled_data)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
