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
from NeuroControl.custom_functions.utils import plot_and_save
import copy


torch.set_printoptions(threshold=64)

# Set up the experiment folder
print("Setting up experiment folder...")
folder_name = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
os.makedirs(folder_name, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Environment setup
sequence_length = 8
frames_per_obs = 8 # Number of frames to stack for input to the agent

image_latent_size_sqrt = 32

agent = NeuralAgent(num_neurons=16, frames_per_step=frames_per_obs, h_state_latent_size=512, image_latent_size_sqrt=image_latent_size_sqrt).to(device)
agent.save_str_file_arch(folder_name + "pre_trained_agent.txt")


env = CarEnv(agent, render_mode=None, device=device, frames_per_obs=frames_per_obs, seq_length=sequence_length)
env.start_data_generation()

env_warmup_time = 1200
print(f"Generating {env_warmup_time} real seconds worth of data")
time.sleep(env_warmup_time)

# Initialize the agent

# agent.pre_training_loss([torch.rand((1,8,96,96)).to(device)],[torch.rand((1,5,)).to(device) > 0.5], [torch.rand((1,8,)).to(device)])
# print("Done.")

# Pre-training loop
# Pre-training loop

num_epochs = 1024#*4#*8#*5#512*2*20#*12#*14
batch_size = 16
losses = []  # List to store loss values

rep_losses = []
predict_losses = []
kl_losses = []
critic_losses = []
actor_losses = []
total_imagined_rewards = []
imagined_actions = []


for epoch in tqdm(range(num_epochs)):
    obs_batch, action_batch, reward_batch = env.sample_buffer(batch_size)
    
    # Convert numpy arrays to PyTorch tensors
    obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(device)
    action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32).to(device)
    reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(device)
    
    world_loss, representation_loss, reward_prediction_loss, kl_loss, mse_rewards_loss, critic_loss_replay, init_lats = agent.replay_training_loss(obs_batch, action_batch, reward_batch, all_losses=True)
    #agent.update_world_model(loss)

    init_h_state = init_lats[1].detach()
    init_obs_lat = init_lats[0].detach()

    actor_loss, critic_loss_imagi, total_imagined_reward, imagined_probable_action = agent.imaginary_training(init_h_state, init_obs_lat, torch.zeros((batch_size, frames_per_obs)))
    
    critic_loss = (0*critic_loss_imagi) + critic_loss_replay

    agent.update_models(world_loss, actor_loss, critic_loss)


    
    losses.append(world_loss.item())  # Store the loss value
    rep_losses.append(representation_loss.item())
    predict_losses.append(reward_prediction_loss.item())
    kl_losses.append(kl_loss.item())
    critic_losses.append(critic_loss.item())
    actor_losses.append(actor_loss.item())
    total_imagined_rewards.append(total_imagined_reward.item())
    imagined_actions.append(imagined_probable_action.item())



    tqdm.write("--------------------------------------------------")
    tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {world_loss.item()}")
    tqdm.write(f"Representation Loss: {representation_loss.item()}")
    tqdm.write(f"MSE Reward Prediction Loss: {mse_rewards_loss.item()}")
    tqdm.write(f"Reward Prediction Loss: {reward_prediction_loss.item()}")
    tqdm.write(f"KL Loss: {kl_loss.item()}")
    tqdm.write(f"Critic Loss: {critic_loss.item()}")
    tqdm.write("Obs Gen/s: " + str(env.current_obs_added_per_s))
    tqdm.write(f"Actor Loss: {actor_loss.item()}")
    tqdm.write(f"Imagined Reward: {total_imagined_reward.item()}")
    tqdm.write(f"Imagined Action: {imagined_probable_action.item()}")



plot_and_save(losses, 'Pre-training Loss', 'Loss', f"{folder_name}/total_loss_graph.png")
plot_and_save(rep_losses, 'Representation Loss', 'Loss', f"{folder_name}/representation_loss_graph.png")
plot_and_save(predict_losses, 'Prediction Loss', 'Loss', f"{folder_name}/prediction_loss_graph.png")
plot_and_save(kl_losses, 'KL Loss', 'Loss', f"{folder_name}/kl_loss_graph.png")
plot_and_save(critic_losses, 'Critic Loss', 'Loss', f"{folder_name}/critic_loss_graph.png")
plot_and_save(actor_losses, 'Actor Loss', 'Loss', f"{folder_name}/actor_loss_graph.png")
plot_and_save(total_imagined_rewards, 'Imagined Reward', 'Reward', f"{folder_name}/imagined_reward_graph.png")
plot_and_save(imagined_actions, 'Imagined Action', 'Action', f"{folder_name}/imagined_action_graph.png")
plot_and_save(copy.deepcopy(env.all_rewards), 'Real Reward', 'Reward', f"{folder_name}/real_reward_graph.png", xlabel="step")

print(f"Loss graphs saved in {folder_name}")


# Create a CSV file to store the losses
csv_filename = f"{folder_name}/losses.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['Batch', 'Total Loss', 'Representation Loss', 'Prediction Loss', 'KL Loss', 'Critic Loss', 'Actor Loss'])
    
    # Write the data
    for i, (total, rep, pred, kl, cri, act) in enumerate(zip(losses, rep_losses, predict_losses, kl_losses, critic_losses, actor_losses)):
        csvwriter.writerow([i+1, total, rep, pred, kl, cri, act])

print(f"Losses saved to {csv_filename}")


agent.save_checkpoint(folder_name + "pre_trained_agent.pth")


print("Generating demo videos...")
n_vids = 3  # Number of videos to generate

for vid_num in range(n_vids):
    with torch.no_grad():
        # Get an initial observation
        obs_ori, _, _ = env.sample_buffer(1)
        obs, actions, _ = env.sample_episodes(1)
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        obs = torch.squeeze(obs, dim=0)
        #obs = obs.view(obs.size(0)*obs.size(1), obs.size(2), obs.size(3))
        #pdb.set_trace()

        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        #pdb.set_trace()
        #actions = torch.squeeze(actions, dim=0)
        actions = actions.view(1, actions.size(1)*actions.size(2), actions.size(3))
        initial_obs = torch.unsqueeze(obs[0, 0:frames_per_obs], dim=0)
        #pdb.set_trace()

        init_h_state = torch.zeros(1, agent.h_state_latent_size).to(device)
        
        # Encode the initial observation
        #pdb.set_trace()
        initial_latent = agent.world_model.encode_obs(initial_obs, init_h_state)
        
        # Predict future latents
        future_steps = 16*8
        #pdb.set_trace()
        #pdb.set_trace()
        predicted_latents, saved_h_states = agent.predict_image_latents(future_steps, initial_latent, actions=actions)
        
        # Decode the predicted latents to observations
        predicted_latents = predicted_latents.view(future_steps, frames_per_obs, (image_latent_size_sqrt**2))
        saved_h_states = saved_h_states.view(future_steps, agent.h_state_latent_size)
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

# Just to watch newest episode
with torch.no_grad():
    obs_batch, action_batch, reward_batch = env.get_latest_episode()
    env.gen_vid_from_obs(obs_batch, f"{folder_name}/real_last_ep.mp4", fps=1.0, frame_size=(96, 96))