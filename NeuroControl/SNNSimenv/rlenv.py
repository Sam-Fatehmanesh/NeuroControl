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
        self.pause_generation = threading.Event()
        self.simulation_thread = None
        self.sim = None

    def simulation_worker(self):
        import nest
        from NeuroControl.SNNSimenv.snnenv import snnEnv

        print("Initializing NEST backend with reduced verbosity.")
        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")

        self.sim = snnEnv(snn_params=self.snn_params, 
                     neuron_params=self.neuron_params, 
                     rl_params=self.rl_params, 
                     snn_filename=None,
                     apply_optical_error=False)

        obs, _ = self.sim.reset()
        
        while not self.stop_generation:
            self.pause_generation.wait()  # Wait if paused

            action = np.random.rand(*self.action_dims) > self.probabilityOfSpikeAction
            obs, reward, done, _ = self.sim.step(action)

            if self.data_queue.full():
                self.data_queue.get()
            self.data_queue.put((obs, action, reward))

            if self.data_queue.qsize() % 16 == 0:
                self.sim.cleanSpikeRecorder()

            if done:
                obs, _ = self.sim.reset()

    def start_data_generation(self):
        self.stop_generation = False
        self.pause_generation.set()  # Ensure it starts unpaused
        self.simulation_thread = threading.Thread(target=self.simulation_worker)
        self.simulation_thread.start()

    def stop_data_generation(self):
        self.stop_generation = True
        self.pause_generation.set()  # Ensure it's not paused when stopping
        if self.simulation_thread:
            self.simulation_thread.join()

    def pause_simulation(self):
        self.pause_generation.clear()

    def resume_simulation(self):
        self.pause_generation.set()

    def sample_buffer(self, batch_size):
        self.pause_simulation()  # Pause the simulator

        if self.data_queue.qsize() < batch_size:
            self.resume_simulation()  # Resume if not enough data
            return None

        sampled_data = random.sample(list(self.data_queue.queue), batch_size)
        
        obs_batch, action_batch, reward_batch = zip(*sampled_data)

        self.resume_simulation()  # Resume the simulator

        return obs_batch, action_batch, reward_batch
