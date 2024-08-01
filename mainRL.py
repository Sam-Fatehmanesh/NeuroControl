import numpy as np
from NeuroControl.SNNSimenv.snnenv import snnEnv
from NeuroControl.SNNSimenv.synthCI import create_video
from NeuroControl.models.world import *#WorldModelT, WorldModelNO
from NeuroControl.models.actor import NeuronControlActor
from NeuroControl.models.critic import NeuralControlCritic

from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import nest
from tqdm import tqdm
import cv2
import pdb
import os
from matplotlib import pyplot as plt
import csv
import time

# Checks if a folder called experiments exists if not it makes it
print("Checking if 'experiments' folder exists.")
if not os.path.exists('experiments'):
    os.makedirs('experiments')
    print("'experiments' folder created.")

# Creates a folder in it with a filename set by datetime.now()
print("Creating a folder for the current experiment.")
folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
folder_name = f'experiments/{folder_name}'
os.makedirs(folder_name)
folder_name += "/"

print("Initializing NEST backend with reduced verbosity.")
nest.set_verbosity(30)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# RL Parameters
rl_params = {
    "steps_per_ep": 16,  # Total number of steps per episode
    "score_factor": 0.1   # Scoring factor for rewards
}

# SNN Parameters
num_neurons = 16
snn_params = {
    "num_neurons": num_neurons,
    "inhibitory_exist": True,
    "fraction_inhibitory": 0.5,
    "step_action_observsation_simulation_time": 8,
    "noise_rate": 0,
    "neuron_connection_probability": 0.2,
    "synapse_delay_time_length": 1.0,
    "synapse_weight_factor": 1,
    "noise_weight": 1.0,
    "fraction_stimulated": 0.2,
    "stimulation_probability": 1,
    "stimulator_synapse_weight": 3000,
    "stimulation_time_resolution": 0.1,
    "num_recorded_neurons": 10,
    "num_neurons_stimulated": int(0.2*num_neurons),
    "ih_synapse_weight_factor": 1,
    "auto_ih": True,
}

num_stim_neurons = int(snn_params["fraction_stimulated"] * snn_params["num_neurons"])

# Neuron Parameters
neuron_params = {
    "C_m": 0.25,  # nF    membrane capacitance
    "I_e": 0.5,   # nA    bias current
    "tau_m": 20.0,  # ms    membrane time constant
    "t_ref": 2.0,  # ms    refractory period
    "tau_syn_ex": 5.0,  # ms    excitatory synapse time constant
    "tau_syn_in": 5.0,  # ms    inhibitory synapse time constant
    "V_reset": -70.0,  # mV    reset membrane potential
    "E_L": -65.0,  # mV    resting membrane potential
    "V_th": -50.0  # mV    firing threshold voltage
}

# Set random seed
print("Setting random seeds.")
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

print("Initializing the environment.")
env = snnEnv(snn_params=snn_params, 
neuron_params=neuron_params, 
rl_params=rl_params, 
snn_filename=None,
apply_optical_error=False)
action_size = int(snn_params["num_neurons_stimulated"] * env.step_action_observsation_simulation_time)
action_dims = (snn_params["num_neurons_stimulated"], env.step_action_observsation_simulation_time)

image_n = 280
num_frames_per_step = snn_params["step_action_observsation_simulation_time"]
latent_size =int(32**2)
state_size = latent_size

probabilityOfSpikeAction = 0.8

print("Initializing models.")
world_model = WorldModelMamba(image_n, num_frames_per_step, latent_size, state_size, action_size).to(device)
actor_model = NeuralControlActor(state_size, action_dims).to(device)
critic_model = NeuralControlCritic(state_size, latent_size).to(device)

# Loss function and optimizer
print("Setting up optimizers.")
optimizer_w = optim.Adam(world_model.parameters(), lr=0.0001)
optimizer_a = optim.Adam(actor_model.parameters(), lr=0.0001)
optimizer_c = optim.Adam(critic_model.parameters(), lr=0.0001)


losses_w = []
losses_a = []
losses_c = []

num_episodes = 8096

# Saves a text file with the str of the model and the saved parameter dictionaries
print("Saving model and parameter configurations.")
with open(folder_name + 'params.txt', 'w') as f:
    f.write(str(world_model))
    f.write('\n\n')
    f.write(("snn_params"))
    f.write(str(snn_params))
    f.write('\n\n')
    f.write(("neuron_params"))
    f.write(str(neuron_params))
    f.write('\n\n')
    f.write(("rl_params"))
    f.write(str(rl_params))
    f.write('\n\n')
    f.write(("seed"))
    f.write(str(seed))
    f.write('\n\n')
    f.write("num_episodes")
    f.write(str(num_episodes))

# collects and graphs sim step speed times
sim_step_speed_times =[]

# This prediction system will use backprop through time with one update step per episode
# Training loop
print("Starting training loop.")
with tqdm(total=num_episodes, desc="Episodes") as pbar:
    for episode in range(num_episodes):
        # Reset the environment
        first_obs, _ = env.reset()
        total_loss_w = torch.zeros((1)).to(device)
        # Convert state to tensor
        #print(first_obs)
        first_obs = np.array(first_obs, dtype=np.float32)  # Convert list of numpy ndarrays to a single numpy ndarray
        first_obs = torch.tensor(first_obs).unsqueeze(0).to(device) / 255.0
        
        prev_obs = first_obs

        model_state = torch.rand(state_size).to(device)

        action_entropy_sum = torch.zeros((1)).to(device)

        predicted_total_rewards = torch.empty((0)).to(device)

        total_true_reward = np.zeros((1))

        for step in tqdm(range(rl_params["steps_per_ep"]), leave=False):
            t1 = time.time()


            steps_left = rl_params["steps_per_ep"] - step
            # Select action

            #action = np.random.uniform(0.0, 1.0, (snn_params["num_neurons_stimulated"], int(env.step_action_observsation_simulation_time))) < probabilityOfSpikeAction
            action = actor_model(model_state)
            action_entropy_sum += actor_model.entropy(action)

            #action that is fed to the nest sim
            action_sample = torch.bernoulli(action).detach().cpu().numpy()

            # action equals a 2d binary matrix where
            
            # Forward pass
            prev_obs = torch.transpose(prev_obs, 0, 1)
            
            predicted_obs, model_state = world_model(prev_obs, action, model_state)


            steps_left_tensor = steps_left *  torch.ones((1)).to(device)
            current_reward_total_tensor = total_true_reward *  torch.ones((1)).to(device)

            predicted_reward = critic_model(model_state, steps_left_tensor, current_reward_total_tensor)
            predicted_total_rewards = torch.cat((predicted_total_rewards, predicted_reward))



            # Take a step in the environment
            true_obs, reward, done, _ = env.step(action_sample)

            total_true_reward += reward


            # Convert next_state to tensor
            true_obs = np.array(true_obs, dtype=np.float32)  # Convert list of numpy ndarrays to a single numpy ndarray
            true_obs = torch.tensor(true_obs).unsqueeze(0).to(device) / 255.0

            # Compute loss
            predicted_obs = torch.transpose(predicted_obs, 0, 1)
            loss = world_model.loss(predicted_obs, true_obs)
            total_loss_w += loss


            prev_obs = true_obs

            t2 = time.time()

            sim_step_speed_times.append(t2-t1)


            if done:
                break

        actor_entropy_loss_factor = 0.001
        predicted_total_rewards_sum = torch.sum(predicted_total_rewards)
        actor_model_loss = -(predicted_total_rewards_sum + (actor_entropy_loss_factor*action_entropy_sum))
        losses_a.append(actor_model_loss.item())
        optimizer_a.zero_grad()
        actor_model_loss.backward()
        optimizer_a.step()


        critic_model_loss =  critic_model.loss(predicted_total_rewards - total_true_reward)
        losses_c.append(critic_model_loss.item())
        optimizer_c.zero_grad()
        critic_model_loss.backward()
        optimizer_c.step()


        
            
        losses_w.append(total_loss_w.item())

        # Backward pass and optimize
        optimizer_w.zero_grad()
        total_loss_w.backward()
        optimizer_w.step()

        

        #torch.cuda.empty_cache()
        tqdm.write(f"World Model Loss: {total_loss_w.item()}")
        tqdm.write(f"Actor Model Loss: {actor_model_loss.item()}")
        tqdm.write(f"Critic Model Loss: {critic_model_loss.item()}")
        #pbar.set_postfix({"Loss": total_loss_w.item()})

        pbar.update(1)

        #print(f"Episode {episode + 1}, Total Loss: {total_loss_w.item()}")

# print("Saving model checkpoint")
# # Save the model
# checkpoint = {
#     'model_state_dict': world_model.state_dict(),
# }
# torch.save(checkpoint, folder_name+"checkpoint.pth")

# Runs one episode of the same number of steps and then saves a video of the 
# true observations by running env.render() and saves a video of what the model predicted
print("Generating example prediction video")
world_model.eval()
actor_model.eval()
critic_model.eval()


first_obs, _ = env.reset()
total_loss_w = torch.zeros((1)).to(device)
# Convert state to tensor
#print(first_obs)
first_obs = np.array(first_obs, dtype=np.float32)  # Convert list of numpy ndarrays to a single numpy ndarray
first_obs = torch.tensor(first_obs).unsqueeze(0).to(device) / 255.0

prev_obs = first_obs

model_state = torch.rand(state_size).to(device)

action_entropy_sum = torch.zeros((1)).to(device)

predicted_total_rewards = torch.empty((0)).to(device)

total_true_reward = np.zeros((1))

predictions = []

for step in tqdm(range(rl_params["steps_per_ep"]), leave=False):
    t1 = time.time()


    steps_left = rl_params["steps_per_ep"] - step
    # Select action

    #action = np.random.uniform(0.0, 1.0, (snn_params["num_neurons_stimulated"], int(env.step_action_observsation_simulation_time))) < probabilityOfSpikeAction
    action = actor_model(model_state)
    action_entropy_sum += actor_model.entropy(action)

    #action that is fed to the nest sim
    action_sample = torch.bernoulli(action).detach().cpu().numpy()

    # action equals a 2d binary matrix where
    
    # Forward pass
    prev_obs = torch.transpose(prev_obs, 0, 1)
    
    predicted_obs, model_state = world_model(prev_obs, action, model_state)





    predicted_obs_np = predicted_obs.detach().cpu().numpy()

    # Split the numpy array into multiple frames and multiply by 255.0 to get the pixel values
    #pdb.set_trace()
    predicted_obs_np= np.squeeze(predicted_obs_np, axis=1)
    predicted_obs_np = [frame for frame in predicted_obs_np]
    #pdb.set_trace()

    # Add each frame to the predictions list
    predictions.extend(predicted_obs_np)





    steps_left_tensor = steps_left *  torch.ones((1)).to(device)
    current_reward_total_tensor = total_true_reward *  torch.ones((1)).to(device)

    predicted_reward = critic_model(model_state, steps_left_tensor, current_reward_total_tensor)
    predicted_total_rewards = torch.cat((predicted_total_rewards, predicted_reward))



    # Take a step in the environment
    true_obs, reward, done, _ = env.step(action_sample)

    total_true_reward += reward


    # Convert next_state to tensor
    true_obs = np.array(true_obs, dtype=np.float32)  # Convert list of numpy ndarrays to a single numpy ndarray
    true_obs = torch.tensor(true_obs).unsqueeze(0).to(device) / 255.0

    # Compute loss
    predicted_obs = torch.transpose(predicted_obs, 0, 1)
    loss = world_model.loss(predicted_obs, true_obs)
    total_loss_w += loss


    prev_obs = true_obs

    t2 = time.time()

    sim_step_speed_times.append(t2-t1)






# Saves a video render of the true simulation
sim_steps_rendering = snn_params["step_action_observsation_simulation_time"] * rl_params["steps_per_ep"]
env.render(sim_steps_rendering, save_dir=folder_name)

# Saves a video render of the predicted simulation from predictions list of numpy files into an mp4 not using env

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 1.0
frame_size = (280, 280)  # Assuming the frames are square with edge length `image_n`
filename = folder_name + "predicted_simulation.mp4"
out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

#pdb.set_trace()
# Write each frame to the video
#pdb.set_trace()
for frame in predictions:
    frame *= 255.0
    # Convert the frame from floating-point to 8-bit unsigned integer
    frame = frame.squeeze()  # Remove singleton dimensions if any
    #pdb.set_trace()
    #print(frame)
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        # If the frame is grayscale, convert it to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Write the frame to the video
    out.write(frame)

# Release the VideoWriter object
out.release()

# print(sim_steps_rendering)
# print(len(predictions))

# Saves a graph of the loss over time
plt.plot(losses_w)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("World Model Loss over Time")
plt.savefig(folder_name + "w_loss_graph.png")
plt.close()

# Saves a graph of the loss over time
plt.plot(losses_a)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Actor Model Loss over Time")
plt.savefig(folder_name + "actor_loss_graph.png")
plt.close()


# Saves a graph of the loss over time
plt.plot(losses_c)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Critic Model Loss over Time")
plt.savefig(folder_name + "c_loss_graph.png")
plt.close()

# Saves a csv of the loss over time with each row of the csv being a differnt training episode
# losses is a 1D list of losses_for each episode
episode_index = [str(i) for i in range(num_episodes)]
losses_w_str = [str(loss) for loss in losses_w]
losses_a_str = [str(loss) for loss in losses_a]
losses_c_str = [str(loss) for loss in losses_c]


with open(folder_name + "losses.csv", "w") as f:
    writer = csv.writer(f)
    # Writes the header
    writer.writerow(["Episode", "World Model Loss", "Actor Model Loss", "Critic Model Loss"])
    # Writes each episode and its corresponding loss in a separate row
    for idx, loss_w, loss_a, loss_c in zip(episode_index, losses_w_str, losses_a_str, losses_c_str):
        writer.writerow([idx, loss_w, loss_a, loss_c])

env.close(dirprefix=folder_name)


# Graphs sim_step_speed_times
plt.plot(sim_step_speed_times)
plt.xlabel("Step")
plt.ylabel("Time (s)")
plt.title("Simulation Step Speed over Time")
plt.savefig(folder_name + "sim_step_speed_graph.png")

print(folder_name)

