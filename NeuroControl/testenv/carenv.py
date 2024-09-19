import numpy as np
import torch
import os
import random
import threading
import queue
import cv2
import gymnasium as gym
import time
import pdb

class CarEnv:
    def __init__(self, agent, render_mode=None, device='cpu', frames_per_obs=10, seq_length=32, continuous=False):
        self.device = device
        self.env = None
        self.continuous = continuous
        self.agent = agent
        if continuous:
            self.env = gym.make("CarRacing-v2", render_mode=render_mode, continuous=True)
        else:
            self.env = gym.make("CarRacing-v2", render_mode=render_mode, continuous=False)

        self.frames_per_obs = frames_per_obs
        self.batch_length = self.frames_per_obs *  seq_length
        
        print("Setting random seeds.")
        seed = 42
        np.random.seed(seed)
        torch.set_default_device(device)
        torch.manual_seed(seed)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        if continuous:
            self.action_size = 3
            self.action_dims = (3,)
        else:
            self.action_size = 5  # Discrete CarRacing has 5 actions
            self.action_dims = (5,)
        

        self.data_buffer = queue.Queue(maxsize=1000)  # Reduced max size due to full episodes
        self.stop_generation = False
        self.pause_generation = threading.Event()
        self.simulation_thread = None

        self.current_obs_added_per_s = 1
        self.all_rewards = []

    def preprocess_observation(self, obs_sequence):
        processed_sequence = []
        for obs in obs_sequence:
            processed_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            processed_sequence.append(processed_obs)
        return np.array(processed_sequence)



        
    def simulation_worker(self):
        while not self.stop_generation:
            self.pause_generation.wait()  # Wait if paused

            start_time = time.time()

            obs, _ = self.env.reset()
            
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            
            terminated = False
            truncated = False

            obs_sequence = []
            action_sequence = []
            reward_sequence = []

            hidden_state = torch.zeros(1, self.agent.h_state_latent_size).to(self.device)
            obs_lat = torch.zeros(1, self.agent.seq_obs_latent).to(self.device)

            steps_in_ep_sofar = 0
            action_seq = None
            action_seq_index = 0

            while (not (terminated or truncated)):
                if steps_in_ep_sofar < self.frames_per_obs * 8:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        if action_seq_index > self.frames_per_obs-1 or action_seq_index == 0:
                            # Get the last observation from the episode
                            obs = torch.tensor(np.array(episode_obs[-1])/255.0, dtype=torch.float32).to(self.device)[-self.frames_per_obs:].unsqueeze(0)
                            actions = torch.tensor(np.array(episode_actions[-1]), dtype=torch.float32).to(self.device)[-self.frames_per_obs:].unsqueeze(0)

                            #pdb.set_trace()

                            hidden_state, obs_lat = self.agent.state_and_obslat_from_obs(obs, actions, hidden_state)



                            action_seq = self.agent.act_sample_from_hidden_state_and_obs(hidden_state, obs_lat)

                            action_seq_index = 0
                            action = torch.argmax(action_seq[0, action_seq_index]).item()    
                        else:
                            #pdb.set_trace()
                            action = torch.argmax(action_seq[0, action_seq_index]).item()
                        
                    action_seq_index += 1


                # action = self.env.action_space.sample()


                
                # if self.continuous:
                #     action[1] = 1.0
                #     action[2] = 0.0
                # else:
                #     action = 3  # Always go straight
                # # Turn action from discrete integer into one hot

                #print(action)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                obs_sequence.append(next_obs)
                
                action = np.array(action)
                if not self.continuous:
                    action = np.eye(self.action_size)[action]

                action_sequence.append(action)
                reward_sequence.append(reward)

                self.all_rewards.append(reward)

                if len(obs_sequence) == self.batch_length:
                    episode_obs.append(self.preprocess_observation(obs_sequence))
                    episode_actions.append(action_sequence)
                    episode_rewards.append(reward_sequence)
                    obs_sequence = []
                    action_sequence = []
                    reward_sequence = []
                
                steps_in_ep_sofar += 1
                    

            # Add any remaining partial sequence
            if obs_sequence:
                while len(obs_sequence) < self.batch_length:
                    obs_sequence.append(next_obs)
                    action_sequence.append(np.zeros(self.action_size))
                    reward_sequence.append(0)  # Pad with zero rewards
                episode_obs.append(255 * np.random.rand(*(np.array(self.preprocess_observation(obs_sequence)).shape)))
                episode_actions.append(action_sequence)
                episode_rewards.append(reward_sequence)

            if self.data_buffer.full():
                self.data_buffer.get()
            self.data_buffer.put((episode_obs, episode_actions, episode_rewards))

            end_time = time.time()
            time_taken = end_time - start_time
            self.current_obs_added_per_s = 8*len(episode_obs)/ time_taken  # Episodes per second
            


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
        list_data_buffer = list(self.data_buffer.queue)
        self.resume_simulation()

        if self.data_buffer.qsize() < 1:
            self.resume_simulation()  # Resume if not enough data
            print("Not enough data!")
            return None

        obs_batch, action_batch, reward_batch = [], [], []

        for _ in range(batch_size):
            episode = random.choice(list_data_buffer)
            episode_obs, episode_actions, episode_rewards = episode

            if len(episode_obs) < 1:
                continue  # Skip episodes that are too short

            idx = random.randint(0, len(episode_obs) - 1)

            obs_batch.append(episode_obs[idx])
            action_batch.append(episode_actions[idx])
            reward_batch.append(episode_rewards[idx])

        #self.resume_simulation()  # Resume the simulator

        # Normalize obs_batch to be between 0-1
        #pdb.set_trace()
        obs_batch = np.array(obs_batch) / 255.0

        return obs_batch, action_batch, reward_batch

    def sample_episodes(self, num_episodes):
        self.pause_simulation()  # Pause the simulator

        if self.data_buffer.qsize() < num_episodes:
            self.resume_simulation()  # Resume if not enough data
            return None
        
        sampled_episodes = random.sample(list(self.data_buffer.queue), num_episodes)

        self.resume_simulation()
        #pdb.set_trace()
        obs_batch, action_batch, reward_batch = zip(*sampled_episodes)

          # Resume the simulator

        # Normalize obs_batch to be between 0-1

        obs_batch = np.array(obs_batch) / 255.0

        return obs_batch, np.array(action_batch), np.array(reward_batch)

    def get_episode_with_index(self, index):
        self.pause_simulation()  # Pause the simulator

        # Check if the index is valid
        if index < 0 or index >= self.data_buffer.qsize():
            self.resume_simulation()  # Resume if index is invalid
            return None
        
        # Get the episode at the specified index
        episode = list(self.data_buffer.queue)[index]

        # Resume the simulator
        self.resume_simulation()

        # Normalize obs_batch to be between 0-1
        obs_batch, action_batch, reward_batch = episode
        # Normalize obs_batch to be between 0-1
        obs_batch = np.array(obs_batch) / 255.0

        return obs_batch, np.array(action_batch), np.array(reward_batch)

    def get_latest_episode(self):        
        # Get the latest episode
        return self.get_episode_with_index(self.data_buffer.qsize() - 1)



        

    def gen_vid_from_obs(self, obs, filename="simulation.mp4", fps=10.0, frame_size=(96, 96)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

        for frame_sequence in obs:
            for frame in frame_sequence:
                frame = frame * 255.0
                frame = np.clip(frame, 0.0, 255.0)
                
                frame = np.squeeze(frame)
                
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                if frame.shape[:2] != frame_size:
                    frame = cv2.resize(frame, frame_size)
                
                out.write(frame)

        out.release()

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs_sequence = [obs] * self.frames_per_obs  # Repeat the initial observation
        return self.preprocess_observation(obs_sequence), info

    def step(self, action):
        obs_sequence = []
        total_reward = 0
        terminated = False
        truncated = False
        info = None

        for _ in range(self.frames_per_obs):
            obs, reward, term, trunc, info = self.env.step(action)
            obs_sequence.append(obs)
            total_reward += reward
            terminated |= term
            truncated |= trunc
            if terminated or truncated:
                break

        # Pad the sequence if it's shorter than seq_length
        while len(obs_sequence) < self.frames_per_obs:
            obs_sequence.append(obs)

        return self.preprocess_observation(obs_sequence), total_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
