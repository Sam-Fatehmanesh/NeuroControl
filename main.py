import numpy as np
from LFNeuroControl.SNNSimenv.snnenv import snnEnv
from LFNeuroControl.SNNSimenv.synthCI import create_video
from LFNeuroControl.models.world import WorldModelT

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

# Checks if a folder called experiments exists if not it makes it
if not os.path.exists('experiments'):
    os.makedirs('experiments')

# Creates a folder in it with a filename set by datetime.now()
folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
folder_name = f'experiments/{folder_name}'
os.makedirs(folder_name)
folder_name += "/"

nest.set_verbosity(30)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# RL Parameters
rl_params = {
    "steps_per_ep": 16,  # Total number of steps per episode
    "score_factor": 0.1   # Scoring factor for rewards
}

# SNN Parameters

snn_params = {
    "num_neurons": 16,
    "inhibitory_exist": True,
    "fraction_inhibitory": 0.5,
    "step_action_observsation_simulation_time": 8,
    "noise_rate": 800,
    "neuron_connection_probability": 0.2,
    "synapse_delay_time_length": 1.0,
    "synapse_weight_factor": 1,
    "noise_weight": 1.0,
    "fraction_stimulated": 0.2,
    "stimulation_probability": 1,
    "stimulator_synapse_weight": 1.3,
    "stimulation_time_resolution": 0.1,
    "num_recorded_neurons": 10,
    "num_neurons_stimulated": int(0.2*1024),
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
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


env = snnEnv(snn_params=snn_params, 
neuron_params=neuron_params, 
rl_params=rl_params, 
snn_filename=None)
#env.step(np.ones((snn_params["num_neurons_stimulated"], int(env.step_action_observsation_simulation_time))))


image_n = 280
num_frames_per_step = snn_params["step_action_observsation_simulation_time"]
latent_size = 256
state_size = 256

# World Model
world_model = WorldModelT(image_n, num_frames_per_step, latent_size, state_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(world_model.parameters(), lr=0.00001)

losses = []
num_episodes = 2048

# Saves a text file with the str of the model and the saved parameter dictionaries
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


# This prediction system will use backprop through time with one update step per episode
# Training loop
with tqdm(total=num_episodes, desc="Episodes") as pbar:
    for episode in range(num_episodes):
        # Reset the environment
        first_obs, _ = env.reset()
        total_loss = torch.zeros((1)).to(device)
        # Convert state to tensor
        #print(first_obs)
        first_obs = torch.tensor(first_obs, dtype=torch.float32).unsqueeze(0).to(device)  / 255.0
        prev_obs = first_obs

        model_state = torch.zeros(state_size).to(device)


        for step in tqdm(range(rl_params["steps_per_ep"]), leave=False):
            # Select action
            action = np.zeros((snn_params["num_neurons_stimulated"], int(env.step_action_observsation_simulation_time))) #env.action_space.sample()
            
            
            # Forward pass
            prev_obs = torch.transpose(prev_obs, 0, 1)
            
            predicted_obs, model_state = world_model(prev_obs, model_state)
        


            # Take a step in the environment
            true_obs, reward, done, _ = env.step(action)


            # Convert next_state to tensor
            true_obs = torch.tensor(true_obs, dtype=torch.float32).unsqueeze(0).to(device)  / 255.0


            # Compute loss
            predicted_obs = torch.transpose(predicted_obs, 0, 1)
            loss = criterion(predicted_obs, true_obs)
            total_loss += loss


            prev_obs = true_obs


            if done:
                break
        # Backward pass and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

        pbar.set_postfix({"Loss": total_loss.item()})
        pbar.update(1)
        #print(f"Episode {episode + 1}, Total Loss: {total_loss.item()}")

# Runs one episode of the same number of steps and then saves a video of the 
# true observations by running env.render() and saves a video of what the model predicted
world_model.eval()
# Reset the environment
predictions = []
for step in tqdm(range(rl_params["steps_per_ep"]), leave=False):
    # Select action
    action = np.zeros((snn_params["num_neurons_stimulated"], int(env.step_action_observsation_simulation_time))) #env.action_space.sample()
    
    
    # Forward pass
    prev_obs = torch.transpose(prev_obs, 0, 1)
    
    #pdb.set_trace()
    predicted_obs, model_state = world_model(prev_obs, model_state)
    
    # Since each predicted obs is multiple frames it needs to be split thus
    # Now the predicted_obs is split into a list of numpy arrays and this is added to predictions list
    # Convert the tensor to a numpy array
    predicted_obs_np = predicted_obs.detach().cpu().numpy()

    # Split the numpy array into multiple frames and multiply by 255.0 to get the pixel values
    #pdb.set_trace()
    predicted_obs_np= np.squeeze(predicted_obs_np, axis=1)
    predicted_obs_np = [frame for frame in predicted_obs_np]
    #pdb.set_trace()

    # Add each frame to the predictions list
    predictions.extend(predicted_obs_np)


    # Take a step in the environment
    true_obs, reward, done, _ = env.step(action)
    #pdb.set_trace()


    # Convert next_state to tensor
    true_obs = torch.tensor(true_obs, dtype=torch.float32).unsqueeze(0).to(device) / 255.0


    # Compute loss
    predicted_obs = torch.transpose(predicted_obs, 0, 1)
    loss = criterion(predicted_obs, true_obs)
    total_loss += loss


    prev_obs = true_obs

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
plt.plot(losses)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Loss over Time")
plt.savefig(folder_name + "loss_graph.png")
plt.close()