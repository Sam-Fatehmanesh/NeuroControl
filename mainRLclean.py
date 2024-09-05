import torch
import time
from NeuroControl.testenv.carenv import CarEnv
from NeuroControl.control.agent import NeuralAgent
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import csv

# Set up the experiment folder
print("Setting up experiment folder...")
folder_name = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
os.makedirs(folder_name, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Environment setup
sequence_length = 8
env = CarEnv(render_mode=None, device=device, sequence_length=sequence_length)
env.start_data_generation()

print("Generating 10 real seconds worth of data")
time.sleep(10)

# Initialize the agent
agent = NeuralAgent(num_neurons=16, frames_per_step=sequence_length, state_latent_size=256, steps_per_ep=8, env=env)

# agent.pre_training_loss([torch.rand((1,8,96,96)).to(device)],[torch.rand((1,5,)).to(device) > 0.5], [torch.rand((1,8,)).to(device)])
# print("Done.")

# Pre-training loop
# Pre-training loop
num_epochs = 512#*8#*14
batch_size = 8
losses = []  # List to store loss values

rep_losses = []
predict_losses = []
kl_losses = []

for epoch in tqdm(range(num_epochs)):
    obs_batch, action_batch, reward_batch = env.sample_buffer(batch_size)
    
    # Convert numpy arrays to PyTorch tensors
    obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(device)
    action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32).to(device)
    reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(device)
    
    loss, representation_loss, reward_prediction_loss, kl_loss = agent.pre_training_loss(obs_batch, action_batch, reward_batch, all_losses=True)
    
    agent.optimizer_w.zero_grad()
    loss.backward()
    agent.optimizer_w.step()
    
    losses.append(loss.item())  # Store the loss value
    rep_losses.append(representation_loss.item())
    predict_losses.append(reward_prediction_loss.item())
    kl_losses.append(kl_loss.item())
    
    tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Generate and save the loss graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses)
plt.title('Pre-training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(f"{folder_name}/loss_graph.png")
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), rep_losses, label='Representation Loss')
plt.title('Pre-training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(f"{folder_name}/Representation_loss_graph.png")
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), predict_losses)
plt.title('Pre-training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(f"{folder_name}/prediction_loss_graph.png")
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), kl_losses)
plt.title('Pre-training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(f"{folder_name}/kl_loss_graph.png")
plt.close()

print(f"Loss graph saved as {folder_name}/total_loss_graph.png")


# Create a CSV file to store the losses
csv_filename = f"{folder_name}/losses.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['Batch', 'Total Loss', 'Representation Loss', 'Prediction Loss', 'KL Loss'])
    
    # Write the data
    for i, (total, rep, pred, kl) in enumerate(zip(losses, rep_losses, predict_losses, kl_losses)):
        csvwriter.writerow([i+1, total, rep, pred, kl])

print(f"Losses saved to {csv_filename}")

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


print("Generating demo videos...")
n_vids = 1  # Number of videos to generate

for vid_num in range(n_vids):
    with torch.no_grad():
        # Get an initial observation
        initial_obs, _, _ = env.sample_buffer(1)
        initial_obs = torch.tensor(initial_obs, dtype=torch.float32).to(device)

        init_h_state = torch.zeros(1, agent.state_latent_size).to(device)
        
        # Encode the initial observation
        initial_latent = agent.world_model.encode_obs(initial_obs, init_h_state)
        
        # Predict future latents
        future_steps = 64
        predicted_latents, saved_h_states = agent.predict_image_latents(future_steps, initial_latent)
        
        # Decode the predicted latents to observations
        predicted_latents = predicted_latents.view(future_steps, 8, (16**2))
        saved_h_states = saved_h_states.view(future_steps, agent.state_latent_size)
        predicted_obs = agent.predict_obs(predicted_latents, saved_h_states)

        
        # Move the predictions to CPU for video generation
        predicted_obs = predicted_obs.cpu().numpy()
        
        # Generate video
        video_filename = f"{folder_name}/predicted_future_{vid_num + 1}.mp4"
        env.gen_vid_from_obs(predicted_obs, video_filename, fps=10.0, frame_size=(96, 96))

    print(f"Demo video {vid_num + 1} generated and saved as {video_filename}")

print(f"All {n_vids} demo videos generated and saved in {folder_name}")

env.stop_data_generation()
print(f"Experiment results saved in {folder_name}")

print("Generating sequential video...")

with torch.no_grad():
    # Initialize the environment
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    hidden_state = torch.zeros(1, agent.state_latent_size).to(device)
    
    # Number of steps to simulate
    num_steps = 200
    
    # Lists to store observations and actions
    all_obs = [obs.squeeze(0).cpu().numpy()]
    all_actions = []
    
    for _ in range(num_steps):
        # Get action from the agent
        action = agent.act(obs, hidden_state)
        action_index = torch.argmax(action).item()  # Convert to discrete action index
        all_actions.append(action_index)
        
        # Step the environment
        next_obs, reward, terminated, truncated, _ = env.step(action_index)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Update hidden state
        _, _, _, hidden_state, _ = agent.world_model(obs, action.unsqueeze(1), hidden_state)
        
        # Store observation
        all_obs.append(next_obs.squeeze(0).cpu().numpy())
        
        # Update current observation
        obs = next_obs
        
        if terminated or truncated:
            break
    
    # Convert lists to numpy arrays
    all_obs = np.array(all_obs)
    all_actions = np.array(all_actions)
    
    # Generate video
    video_filename = f"{folder_name}/sequential_simulation.mp4"
    env.gen_vid_from_obs(all_obs, video_filename, fps=10.0, frame_size=(96, 96))

print(f"Sequential simulation video generated and saved as {video_filename}")

# Optionally, you can also save the actions taken during the simulation
action_filename = f"{folder_name}/sequential_simulation_actions.npy"
np.save(action_filename, all_actions)
print(f"Actions taken during the simulation saved as {action_filename}")

