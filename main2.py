import numpy as np
from LFNeuroControl.SNNSimenv.snnenv import snnEnv
from LFNeuroControl.SNNSimenv.synthCI import create_video
from LFNeuroControl.models.actor import NeuronControlActor
from LFNeuroControl.models.critic import NeuralControlCritic

from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import nest
import time


# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# SNN Parameters   

snn_params = {
    "num_neurons": 16,
    "inhibitory_exist": True,
    "fraction_inhibitory": 0.5,
    "step_action_observsation_simulation_time": 100,
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
time_stim_obs = int(snn_params["step_action_observsation_simulation_time"])
# RL Parameters
rl_params = {
    "steps_per_ep": 64,  # Total number of steps per episode
    "score_factor": 0.1   # Scoring factor for rewards
}

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


env = (snnEnv(snn_params=snn_params, 
neuron_params=neuron_params, 
rl_params=rl_params, 
snn_filename=None))

start_time = time.time()



end_time = time.time()
execution_time = end_time - start_time 




# frames = env.step(np.zeros((snn_params["num_neurons_stimulated"], int(env.step_action_observsation_simulation_time))))[0]
# # frames = [o * 255.0 for o in frames]
# #env.close()
# video_filename = str(datetime.now()) + "_neuron_activity.mp4"
# create_video(frames, filename=video_filename, fps=1)


# Define the model, optimizer, and policy
# state_shape = env.observation_space.shape
# action_shape = env.action_space.shape
# max_action = 1#env.action_space.high[0]

# image_n=280
# num_frames = 8
# actor = NeuronControlActor(num_frames, image_n, 512, 6, num_stim_neurons, snn_params["step_action_observsation_simulation_time"])
# critic1 = NeuralControlCritic(num_stim_neurons, snn_params["step_action_observsation_simulation_time"], image_n, num_frames, 1024, 1024)
# critic2 = NeuralControlCritic(num_stim_neurons, snn_params["step_action_observsation_simulation_time"], image_n, num_frames, 1024, 1024)


# actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
# critic1_optim = optim.Adam(critic1.parameters(), lr=3e-4)
# critic2_optim = optim.Adam(critic2.parameters(), lr=3e-4)


