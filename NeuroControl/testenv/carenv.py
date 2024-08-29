import numpy as np
import torch
import os
import random
import threading
import queue
import cv2
import gymnasium as gym

class CarEnv:
    def __init__(self, render_mode=None, device='cpu', sequence_length=10):
        self.device = device
        self.env = gym.make("CarRacing-v2", render_mode=render_mode)
        self.sequence_length = sequence_length
        
        print("Setting random seeds.")
        seed = 42
        np.random.seed(seed)
        torch.set_default_device(device)
        torch.manual_seed(seed)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.action_size = 3  # CarRacing has 3 continuous actions
        self.action_dims = (3,)

        self.data_buffer = queue.Queue(maxsize=20000)
        self.stop_generation = False
        self.pause_generation = threading.Event()
        self.simulation_thread = None

    def preprocess_observation(self, obs):
        # Convert to grayscale and resize to 280x280
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (280, 280))
        return obs

    def simulation_worker(self):
        obs, _ = self.env.reset()
        obs = self.preprocess_observation(obs)
        
        sequence_obs = []
        sequence_actions = []
        sequence_rewards = []
        
        while not self.stop_generation:
            self.pause_generation.wait()  # Wait if paused

            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.preprocess_observation(next_obs)

            sequence_obs.append(obs)
            sequence_actions.append(action)
            sequence_rewards.append(reward)

            if len(sequence_obs) == self.sequence_length:
                if self.data_buffer.full():
                    self.data_buffer.get()
                self.data_buffer.put((sequence_obs, sequence_actions, sequence_rewards))
                
                # Start a new sequence, but keep the last observation
                sequence_obs = [sequence_obs[-1]]
                sequence_actions = []
                sequence_rewards = []

            obs = next_obs

            if terminated or truncated:
                obs, _ = self.env.reset()
                obs = self.preprocess_observation(obs)
                sequence_obs = []
                sequence_actions = []
                sequence_rewards = []

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

        # Normalize obs_batch to be between 0-1
        obs_batch = np.array(obs_batch) / 255.0

        return obs_batch, action_batch, reward_batch

    def gen_vid_from_obs(self, obs, filename="simulation.mp4", fps=30.0, frame_size=(280, 280)):
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

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.preprocess_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess_observation(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
