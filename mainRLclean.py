import torch
import time
from NeuroControl.SNNSimenv.rlenv import NeuralControl
import pdb

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

env = NeuralControl(snn_params, rl_params, device)
env.neuron_params = neuron_params  # Add this line to pass neuron_params

env.start_data_generation()
time.sleep(2)  # Let it run for 10 seconds
# env.stop_data_generation()
# print("############")
# env.start_data_generation()

# Sample from the buffer
sample = env.sample_buffer(10)
if sample:
    print("Successfully sampled from buffer")
else:
    print("Not enough data in buffer to sample")

print(sample)
# pdb.set_trace()