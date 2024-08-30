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
        self.env = gym.make("CarRacing-v2", render_mode=render_mode, continuous=False)
        self.seq_length = sequence_length
        
        print("Setting random seeds.")
        seed = 42
        np.random.seed(seed)
        torch.set_default_device(device)
        torch.manual_seed(seed)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.action_size = 5  # Discrete CarRacing has 5 actions
        self.action_dims = (1,)

        self.data_buffer = queue.Queue(maxsize=1000)  # Reduced max size due to full episodes
        self.stop_generation = False
        self.pause_generation = threading.Event()
        self.simulation_thread = None

    def preprocess_observation(self, obs):
        # Convert to grayscale and resize to 280x280
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        #obs = cv2.resize(obs, (280, 280))
        return obs

    def simulation_worker(self):
        while not self.stop_generation:
            self.pause_generation.wait()  # Wait if paused

            obs, _ = self.env.reset()
            obs = self.preprocess_observation(obs)
            
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_obs = self.preprocess_observation(next_obs)

                episode_obs.append(obs)
                episode_actions.append(action)
                episode_rewards.append(reward)

                obs = next_obs

            if self.data_buffer.full():
                self.data_buffer.get()
            self.data_buffer.put((episode_obs, episode_actions, episode_rewards))

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

        if self.data_buffer.qsize() < 1:
            self.resume_simulation()  # Resume if not enough data
            print("Not enough data!")
            return None

        obs_batch, action_batch, reward_batch = [], [], []

        for _ in range(batch_size):
            episode = random.choice(list(self.data_buffer.queue))
            episode_obs, episode_actions, episode_rewards = episode

            if len(episode_obs) < self.seq_length:
                continue  # Skip episodes that are too short

            start_idx = random.randint(0, len(episode_obs) - self.seq_length)
            end_idx = start_idx + self.seq_length

            obs_batch.append(episode_obs[start_idx:end_idx])
            action_batch.append(episode_actions[start_idx:end_idx])
            reward_batch.append(episode_rewards[start_idx:end_idx])

        self.resume_simulation()  # Resume the simulator

        # Normalize obs_batch to be between 0-1
        obs_batch = np.array(obs_batch) / 255.0

        return obs_batch, action_batch, reward_batch

    def sample_episodes(self, num_episodes):
        self.pause_simulation()  # Pause the simulator

        if self.data_buffer.qsize() < num_episodes:
            self.resume_simulation()  # Resume if not enough data
            return None

        sampled_episodes = random.sample(list(self.data_buffer.queue), num_episodes)
        
        obs_batch, action_batch, reward_batch = zip(*sampled_episodes)

        self.resume_simulation()  # Resume the simulator

        # Normalize obs_batch to be between 0-1
        obs_batch = [np.array(episode) / 255.0 for episode in obs_batch]

        return obs_batch, action_batch, reward_batch

    def gen_vid_from_obs(self, obs, filename="simulation.mp4", fps=1.0, frame_size=(96, 96)):
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
