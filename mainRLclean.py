import torch
import time
from NeuroControl.testenv.carenv import CarEnv
from NeuroControl.control.agent import NeuralAgent
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Set up the experiment folder
print("Setting up experiment folder...")
folder_name = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
os.makedirs(folder_name, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Environment setup
sequence_length = 8
env = CarEnv(render_mode=None, device=device, sequence_length=sequence_length)
env.start_data_generation()

print("Generating 4 real seconds worth of data")
time.sleep(10)

# Initialize the agent
agent = NeuralAgent(num_neurons=16, frames_per_step=sequence_length, state_latent_size=256, steps_per_ep=8, env=env)

# agent.pre_training_loss([torch.rand((1,8,96,96)).to(device)],[torch.rand((1,5,)).to(device) > 0.5], [torch.rand((1,8,)).to(device)])
# print("Done.")

# Pre-training loop
num_epochs = 100
batch_size = 8

for epoch in range(num_epochs):
    obs_batch, action_batch, reward_batch = env.sample_buffer(batch_size)
    
    # Convert numpy arrays to PyTorch tensors
    obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(device)
    action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32).to(device)
    reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(device)

    #pdb.set_trace()
    
    loss = agent.pre_training_loss(obs_batch, action_batch, reward_batch)
    
    agent.optimizer_w.zero_grad()
    loss.backward()
    agent.optimizer_w.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# # Test the dynamics predictor
# def generate_future_video(agent, initial_obs, num_steps, filename):
#     with torch.no_grad():
#         current_obs = torch.tensor(initial_obs, dtype=torch.float32).unsqueeze(0).to(device)
#         hidden_state = torch.zeros(1, agent.state_latent_size).to(device)
#         action = torch.zeros((1, sequence_length, 5)).to(device)
        
#         future_obs = [current_obs]
        
#         for _ in range(num_steps):
#             decoded_obs, pred_next_obs_lat, _, hidden_state, _ = agent.world_model.forward(current_obs, action, hidden_state)
#             future_obs.append(decoded_obs)
#             current_obs = decoded_obs.view(1, sequence_length, 96, 96)
        
#         future_obs = torch.cat(future_obs, dim=0).cpu().numpy()
#         env.gen_vid_from_obs(future_obs, filename, fps=10.0, frame_size=(96, 96))

# # Generate a test video
# initial_obs, _, _ = env.sample_buffer(1)
# generate_future_video(agent, initial_obs[0], num_steps=50, filename=f"{folder_name}/future_prediction.mp4")

# env.stop_data_generation()
# print(f"Experiment results saved in {folder_name}")

# Generate a demo video
print("Generating demo video...")
with torch.no_grad():
    # Get an initial observation
    initial_obs, _, _ = env.sample_buffer(1)
    initial_obs = torch.tensor(initial_obs, dtype=torch.float32).to(device)
    
    # Encode the initial observation
    initial_latent = agent.world_model.encode_obs(initial_obs)
    
    # Predict future latents
    future_steps = 64
    predicted_latents = agent.predict_image_latents(future_steps, initial_latent)
    
    # Decode the predicted latents to observations
    #predicted_latents = predicted_latents.view(batch_size*future_steps*sequence_length, 256)
    predicted_obs = agent.predict_obs(predicted_latents)
    
    # Move the predictions to CPU for video generation
    predicted_obs = predicted_obs.cpu().numpy()
    
    # Generate video
    env.gen_vid_from_obs(predicted_obs, f"{folder_name}/predicted_future.mp4", fps=10.0, frame_size=(96, 96))

print(f"Demo video generated and saved in {folder_name}/predicted_future.mp4")

env.stop_data_generation()
print(f"Experiment results saved in {folder_name}")