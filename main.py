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
import pdb

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
num_frames_per_step = 8
latent_size = 256
state_size = 256

# World Model
world_model = WorldModelT(image_n, num_frames_per_step, latent_size, state_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(world_model.parameters(), lr=0.001)

# This prediction system will use backprop through time with one update step per episode
# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    # Reset the environment
    first_obs, _ = env.reset()
    total_loss = torch.zeros((1)).to(device)
    # Convert state to tensor
    #print(first_obs)
    first_obs = torch.tensor(first_obs, dtype=torch.float32).unsqueeze(0).to(device)
    prev_obs = first_obs

    model_state = torch.zeros(state_size).to(device)


    for step in tqdm(range(rl_params["steps_per_ep"])):
        # Select action
        action = np.zeros((snn_params["num_neurons_stimulated"], int(env.step_action_observsation_simulation_time))) #env.action_space.sample()
        
        
        # Forward pass
        prev_obs = torch.transpose(prev_obs, 0, 1)
        
        predicted_obs, model_state = world_model(prev_obs, model_state)
    


        # Take a step in the environment
        true_obs, reward, done, _ = env.step(action)


        # Convert next_state to tensor
        true_obs = torch.tensor(true_obs, dtype=torch.float32).unsqueeze(0).to(device)


        # Compute loss
        loss = criterion(predicted_obs, true_obs)
        total_loss += loss


        prev_obs = true_obs


        if done:
            break
    # Backward pass and optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f"Episode {episode + 1}, Total Loss: {total_loss.item()}")