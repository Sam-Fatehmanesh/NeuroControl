import numpy as np
import torch
import os
import random
import threading
import queue
import cv2
import pdb
import time
from collections import deque
from tqdm import tqdm

class NeuralControlEnv:
    def __init__(self, snn_params, neuron_params, rl_params, device):
        self.device = device
        self.snn_params = snn_params
        self.rl_params = rl_params
        self.neuron_params = neuron_params
        
        print("Setting random seeds.")
        seed = 42
        np.random.seed(seed)
        torch.set_default_device(device)
        torch.manual_seed(seed)

        self.action_size = int(snn_params["num_neurons_stimulated"] * snn_params["step_action_observsation_simulation_time"])
        self.action_dims = (snn_params["num_neurons_stimulated"], snn_params["step_action_observsation_simulation_time"])

        self.probabilityOfSpikeAction = 0.5

        self.data_buffer = queue.Queue(maxsize=20000)
        self.stop_generation = False
        self.pause_generation = threading.Event()
        self.simulation_thread = None
        self.sim = None

    
    def simulation_worker(self):
        import nest
        from NeuroControl.SNNSimenv.snnsim import snnSim

        print("Initializing NEST backend with reduced verbosity.")
        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")

        self.sim = snnSim(snn_params=self.snn_params, 
                    neuron_params=self.neuron_params, 
                    rl_params=self.rl_params, 
                    snn_filename=None,
                    apply_optical_error=False)

        obs, _ = self.sim.reset()
        
        # Initialize deque to store timestamps of last 10 additions
        last_10_times = deque(maxlen=10)
        
        while not self.stop_generation:
            self.pause_generation.wait()  # Wait if paused

            action = np.random.rand(*self.action_dims) > self.probabilityOfSpikeAction
            obs, reward, done, _ = self.sim.step(action)

            if self.data_buffer.full():
                self.data_buffer.get()
            self.data_buffer.put((obs, action, reward))

            # Record the timestamp of this addition
            current_time = time.time()
            last_10_times.append(current_time)

            # If we have 10 timestamps, calculate and print the rate
            if len(last_10_times) == 10:
                time_diff = last_10_times[-1] - last_10_times[0]
                rate = 10 / time_diff if time_diff > 0 else 0
                #tqdm.write(f"Data generation rate: {rate:.2f} items/second")

            if self.data_buffer.qsize() % 16 == 0:
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

        if self.data_buffer.qsize() < batch_size:
            self.resume_simulation()  # Resume if not enough data
            return None

        sampled_data = random.sample(list(self.data_buffer.queue), batch_size)
        
        obs_batch, action_batch, reward_batch = zip(*sampled_data)

        self.resume_simulation()  # Resume the simulator

        # Normalizes obs_batch to be between 0-1
        #pdb.set_trace()
        obs_batch = np.array(obs_batch) / 255.0

        return obs_batch, action_batch, reward_batch

    def gen_vid_from_obs(self, obs, filename="simulation.mp4", fps=1.0, frame_size=(280, 280)):
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

        # Write each frame to the video
        for frame in obs:
            # Normalize the frame to 0-255 range
            frame = frame * 255.0
            frame = np.clip(frame, 0.0, 255.0)
            
            # Remove singleton dimensions if any
            frame = np.squeeze(frame)
            
            # Clip values to 0-255 range and convert to 8-bit unsigned integer
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # If the frame is grayscale, convert it to BGR
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Resize the frame if it doesn't match the specified frame_size
            if frame.shape[:2] != frame_size:
                frame = cv2.resize(frame, frame_size)
            
            # Write the frame to the video
            out.write(frame)

        # Release the VideoWriter object
        out.release()

        print(f"Video saved as {filename}")
