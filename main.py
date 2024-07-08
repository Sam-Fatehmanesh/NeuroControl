import numpy as np
from LFNeuroControl.SNNSimenv.snnenv import snnEnv


# SNN Parameters
snn_params = {
    "num_neurons": 64,
    "inhibitory_exist": True,
    "fraction_inhibitory": 0.5,
    "step_action_observsation_simulation_time": 100.0,
    "noise_rate": 800,
    "neuron_connection_probability": 0.1,
    "synapse_delay_time_length": 1.0,
    "synapse_weight_factor": 1,
    "noise_weight": 1.0,
    "fraction_stimulated": 0.2,
    "stimulation_probability": 1,
    "stimulator_synapse_weight": 1.3,
    "stimulation_time_resolution": 0.1,
    "num_recorded_neurons": 10,
    "num_neurons_stimulated": int(0.2*64),
}

# RL Parameters
rl_params = {
    "steps_per_ep": 100,  # Total number of steps per episode
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


env = snnEnv(snn_params=snn_params, 
neuron_params=neuron_params, 
rl_params=rl_params, 
snn_filename="saved.pkl")

env.step(np.ones((snn_params["num_neurons_stimulated"], int(env.step_action_observsation_simulation_time))))
env.close()