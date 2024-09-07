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
sequence_length = 8  # 
frames_per_obs = 8 # Number of frames to stack for input to the agent
env = CarEnv(render_mode=None, device=device, frames_per_obs=frames_per_obs, seq_length=sequence_length)
env.start_data_generation()

print("Generating 10 real seconds worth of data")
time.sleep(10)

image_latent_size_sqrt = 20
# Initialize the agent
agent = NeuralAgent(num_neurons=16, frames_per_step=frames_per_obs, state_latent_size=128*6, steps_per_ep=8, env=env, image_latent_size_sqrt=image_latent_size_sqrt)


# agent.pre_training_loss([torch.rand((1,8,96,96)).to(device)],[torch.rand((1,5,)).to(device) > 0.5], [torch.rand((1,8,)).to(device)])
# print("Done.")

# Pre-training loop
# Pre-training loop
num_epochs = 512*2*20#*12#*14
batch_size = 10
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

    tqdm.write("--------------------------------------------------")
    tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    tqdm.write(f"Representation Loss: {representation_loss.item()}")
    tqdm.write(f"Reward Prediction Loss: {reward_prediction_loss.item()}")
    tqdm.write(f"KL Loss: {kl_loss.item()}")
    tqdm.write("Obs Gen/s: " + str(env.current_obs_added_per_s))

    

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
        obs, _, _ = env.sample_buffer(1)
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        initial_obs = obs[:, 0:frames_per_obs]

        init_h_state = torch.zeros(1, agent.state_latent_size).to(device)
        
        # Encode the initial observation

        initial_latent = agent.world_model.encode_obs(initial_obs, init_h_state)
        
        # Predict future latents
        future_steps = sequence_length
        predicted_latents, saved_h_states = agent.predict_image_latents(future_steps, initial_latent)
        
        # Decode the predicted latents to observations
        predicted_latents = predicted_latents.view(future_steps, frames_per_obs, (image_latent_size_sqrt**2))
        saved_h_states = saved_h_states.view(future_steps, agent.state_latent_size)
        predicted_obs = agent.predict_obs(predicted_latents, saved_h_states)

        
        # Move the predictions to CPU for video generation
        predicted_obs = predicted_obs.cpu().numpy()
        
        # Generate video
        video_filename = f"{folder_name}/predicted_future_{vid_num + 1}.mp4"
        env.gen_vid_from_obs(predicted_obs, video_filename, fps=1.0, frame_size=(96, 96))
        
        obs = obs.cpu().numpy()
        true_video_filename = f"{folder_name}/true_future_{vid_num + 1}.mp4"
        env.gen_vid_from_obs(obs, true_video_filename, fps=1.0, frame_size=(96, 96))



    print(f"Demo video {vid_num + 1} generated and saved as {video_filename}")

print(f"All {n_vids} demo videos generated and saved in {folder_name}")

env.stop_data_generation()
print(f"Experiment results saved in {folder_name}")

print("Generating more demo videos...")
with torch.no_grad():

    obs_batch, action_batch, reward_batch = env.sample_buffer(1)
    obs_orig = obs_batch
    # Convert numpy arrays to PyTorch tensors
    obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(device)
    action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32).to(device)
    reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(device)

    decoded_obs_list = agent.world_model_pre_train_forward(obs_batch, action_batch, reward_batch)


    video_filename = f"{folder_name}/train_style_predicted_future.mp4"
    video_filename_true = f"{folder_name}/true_train_style_predicted_future.mp4"
    
    #pdb.set_trace()
    decoded_obs_list = torch.squeeze(decoded_obs_list, dim=1)
    decoded_obs_list = decoded_obs_list.cpu().numpy()
    
    env.gen_vid_from_obs(decoded_obs_list, video_filename, fps=1.0, frame_size=(96, 96))
    env.gen_vid_from_obs(obs_orig, video_filename_true, fps=1.0, frame_size=(96, 96))



# print("Generating reconstructed video using world_model_pre_train_forward...")

# # Sample a single episode
# sampled_obs, sampled_actions, sampled_rewards = env.sample_episodes(1)

# # Ensure we have a sampled episode
# if sampled_obs is None:
#     print("Failed to sample an episode. Ensure the data buffer is not empty.")
# else:
#     with torch.no_grad():
#         # Convert sampled observations to tensor and move to device
#         obs_tensor = torch.tensor(sampled_obs[0], dtype=torch.float32).to(device)
#         actions_tensor = torch.tensor(sampled_actions[0], dtype=torch.float32).to(device)
#         rewards_tensor = torch.tensor(sampled_rewards[0], dtype=torch.float32).to(device)
        
#         print(f"Obs tensor shape: {obs_tensor.shape}")
#         print(f"Actions tensor shape: {actions_tensor.shape}")
#         print(f"Rewards tensor shape: {rewards_tensor.shape}")
        
#         # Adjust tensor shapes
#         if obs_tensor.shape[1] != agent.frames_per_step:
#             # Reshape obs_tensor to match the expected input shape
#             obs_tensor = obs_tensor.view(-1, agent.frames_per_step, 96, 96)
#         obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension
        
#         if actions_tensor.shape[1] != agent.frames_per_step:
#             # Reshape actions_tensor to match the expected input shape
#             actions_tensor = actions_tensor.view(-1, agent.frames_per_step, 5)
#         actions_tensor = actions_tensor.unsqueeze(0)  # Add batch dimension
        
#         if rewards_tensor.shape[1] != agent.frames_per_step:
#             # Reshape rewards_tensor to match the expected input shape
#             rewards_tensor = rewards_tensor.view(-1, agent.frames_per_step)
#         rewards_tensor = rewards_tensor.unsqueeze(0).unsqueeze(-1)  # Add batch and channel dimensions
        
#         print(f"Adjusted obs tensor shape: {obs_tensor.shape}")
#         print(f"Adjusted actions tensor shape: {actions_tensor.shape}")
#         print(f"Adjusted rewards tensor shape: {rewards_tensor.shape}")
        
#         # Get the decoded observations
#         try:
#             reconstructed_obs_list = agent.world_model_pre_train_forward(obs_tensor, actions_tensor, rewards_tensor)
            
#             # Concatenate the decoded observations
#             reconstructed_obs = torch.cat(reconstructed_obs_list, dim=1).squeeze(0)
            
#             # Move the predictions to CPU for video generation
#             true_obs = obs_tensor.squeeze(0).cpu().numpy()
#             reconstructed_obs = reconstructed_obs.cpu().numpy()
            
#             # Generate videos
#             true_video_filename = f"{folder_name}/true_episode.mp4"
#             reconstructed_video_filename = f"{folder_name}/reconstructed_episode.mp4"
            
#             env.gen_vid_from_obs(true_obs, true_video_filename, fps=10.0, frame_size=(96, 96))
#             env.gen_vid_from_obs(reconstructed_obs, reconstructed_video_filename, fps=10.0, frame_size=(96, 96))

#             print(f"True episode video saved as {true_video_filename}")
#             print(f"Reconstructed episode video saved as {reconstructed_video_filename}")

#         except Exception as e:
#             print(f"An error occurred during reconstruction: {str(e)}")
#             print("Stack trace:")
#             import traceback
#             traceback.print_exc()

#     # Optionally, you can also save the actions from the sampled episode
#     action_filename = f"{folder_name}/sampled_episode_actions.npy"
#     np.save(action_filename, sampled_actions[0])
#     print(f"Actions from the sampled episode saved as {action_filename}")
