import torch
import time
from NeuroControl.SNNSimenv.snnenv import NeuralControlEnv
from NeuroControl.models.autoencoder import NeuralAutoEncoder
import pdb
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# RL Parameters
rl_params = {
    "steps_per_ep": 8,
    "score_factor": 0.1
}

# SNN Parameters
num_neurons = 16
neurons_stimulated_frac = 1.0
snn_params = {
    "num_neurons": num_neurons,
    "inhibitory_exist": True,
    "fraction_inhibitory": 0.5,
    "step_action_observsation_simulation_time": 8,
    "noise_rate": 0,
    "neuron_connection_probability": 1/4,
    "synapse_delay_time_length": 1.0,
    "synapse_weight_factor": 1,
    "noise_weight": 1.0,
    "fraction_stimulated": neurons_stimulated_frac,
    "stimulation_probability": 1,
    "stimulator_synapse_weight": 3000,
    "stimulation_time_resolution": 0.1,
    "num_recorded_neurons": num_neurons,
    "num_neurons_stimulated": int(neurons_stimulated_frac*num_neurons),
    "ih_synapse_weight_factor": 1,
    "auto_ih": True,
}

# Neuron Parameters
neuron_params = {
    "C_m": 0.25,
    "I_e": 0.5,
    "tau_m": 20.0,
    "t_ref": 2.0,
    "tau_syn_ex": 5.0,
    "tau_syn_in": 5.0,
    "V_reset": -70.0,
    "E_L": -65.0,
    "V_th": -50.0
}

env = NeuralControlEnv(snn_params, neuron_params, rl_params, device)
# env.neuron_params = neuron_params  # Add this line to pass neuron_params

env.start_data_generation()
time.sleep(2)  # Let it run for 10 seconds
# print("############")
# env.start_data_generation()

# Sample from the buffer
obs_batch, action_batch, reward_batch = env.sample_buffer(8)
obs_batch = list(obs_batch[0])
# pdb.set_trace()
obs_batch = [ob.astype(float) for ob in obs_batch]
if obs_batch:
    print("Successfully sampled from buffer")
else:
    print("Not enough data in buffer to sample")

print(obs_batch)
# pdb.set_trace()


env.stop_data_generation()


env.gen_vid_from_obs(obs_batch, filename=folder_name+"test.mp4")
ae = NeuralAutoEncoder(8)

obs_batch = torch.unsqueeze(torch.tensor(np.stack(obs_batch)).float(), 0)
# pdb.set_trace()
decoded_obs, lat = ae(obs_batch)

decoded_obs = (decoded_obs.detach().cpu().numpy() * 255)[0]
pdb.set_trace()
env.gen_vid_from_obs(decoded_obs, filename=folder_name+"decoded_test.mp4")
