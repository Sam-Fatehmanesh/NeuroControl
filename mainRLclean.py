import torch
import time
from NeuroControl.SNNSimenv.snnenv import NeuralControlEnv
from NeuroControl.testenv.carenv import CarEnv
from NeuroControl.models.autoencoder import NeuralAutoEncoder
import pdb
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.optim as optim
import cv2
from NeuroControl.custom_functions.utils import STMNsampler, symlog, symexp



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


image_n=96
#env = NeuralControlEnv(snn_params, neuron_params, rl_params, device)
env = CarEnv(render_mode=None, sequence_length=8)
# env.neuron_params = neuron_params  # Add this line to pass neuron_params

env.start_data_generation()
time.sleep(10)  # Let it run for 10 seconds
# print("############")
# env.start_data_generation()

# Sample from the buffer
obs_batch, action_batch, reward_batch = env.sample_buffer(2)
#pdb.set_trace()

obs_batch = list(obs_batch[0])
obs_batch = [ob.astype(float) for ob in obs_batch]
if obs_batch:
    print("Successfully sampled from buffer")
else:
    print("Not enough data in buffer to sample")

print(obs_batch)
# pdb.set_trace()


#env.stop_data_generation()


env.gen_vid_from_obs(obs_batch, filename=folder_name+"test.mp4")
ae = NeuralAutoEncoder(8, image_n)

obs_batch = torch.unsqueeze(torch.tensor(np.stack(obs_batch)).float(), 0)
# pdb.set_trace()
decoded_obs, lat = ae(obs_batch)

decoded_obs = (decoded_obs.detach().cpu().numpy() * 255)[0]
# pdb.set_trace()
env.gen_vid_from_obs(decoded_obs, filename=folder_name+"decoded_test.mp4")



# import torch.optim as optim

# Set up the autoencoder and optimizer
ae = NeuralAutoEncoder(8, image_n).to(device)

optimizer = optim.Adam(ae.parameters(), lr=0.001)

# Training loop
num_epochs = 1
batch_size = 8

batches_per_epoch = 1024

from tqdm import tqdm

# Outer loop for epochs
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    total_loss = 0
    num_batches = 0
    
    # Inner loop for batches within each epoch
    for _ in tqdm(range(batches_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        
        obs_batch, _, _ = env.sample_buffer(batch_size)
        obs_batch = torch.tensor(np.stack(obs_batch)).float().to(device)
        
        

        loss, _ = ae.loss(obs_batch)
        # print("###################")
        # print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        tqdm.write(f"Batch Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Generate demo video
print("Generating demo video...")

# Sample a batch of 32 observations
obs_batch, _, _ = env.sample_buffer(32)
obs_batch = torch.tensor(np.stack(obs_batch)).float().to(device)

# Generate reconstructions
with torch.no_grad():
    decoded_obs, _ = ae(obs_batch)

# Convert to numpy and scale to 0-255 range
true_obs = (obs_batch.cpu().numpy())
decoded_obs = (decoded_obs.cpu().numpy())
# Change both above from 32x8x280x280 to 256x280x280
true_obs = np.transpose(true_obs, (1, 0, 2, 3))
decoded_obs = np.transpose(decoded_obs, (1, 0, 2, 3))
true_obs = np.reshape(true_obs, ( 256, image_n, image_n))
decoded_obs = np.reshape(decoded_obs, ( 256, image_n, image_n))

#decoded_obs = (torch.tensor(decoded_obs)).cpu().numpy()

# pdb.set_trace()

# Generate videos
print("Saving true observations video...")
env.gen_vid_from_obs(true_obs, filename=folder_name+"true_observations.mp4", fps=1)

print("Saving decoded observations video...")
env.gen_vid_from_obs(decoded_obs, filename=folder_name+"decoded_observations.mp4", fps=1)

print("Demo videos generated and saved in the experiment folder.")


print(folder_name)


env.stop_data_generation()