import nest
import gymnasium
from gymnasium import spaces
import numpy as np
import nest
import os
import json

class snnEnv(gymnasium.Env):   
    def __init__(self, snn_params, neuron_params, snn_filename=None):
        super(snnEnv, self).__init__()

        if snn_filename==None:
            # Parameters for the network
            self.num_neurons = snn_params["num_neurons"] # Number of neurons in SNN simulation
            self.inhibitory_exist = snn_params["inhibitory_exist"] # Determins if inhibitory neurons will actually be inhibitory or not.
            self.fraction_inhibitory = snn_params["fraction_inhibitory"]  # Fraction of inhibitory neurons
            self.simulation_time = snn_params["simulation_time"]  # Total length of simulation
            self.noise_rate = snn_params["noise_rate"] # Rate of noise stimulation in hertz
            self.neuron_connection_probability = snn_params["neuron_connection_probability"]  # probability that there exists a connection directed from any one neuron to any other neuron
            self.synapse_delay_time_length = snn_params["synapse_delay_time_length"] # time delay for spike signals for all synapses on all neurons
            self.synapse_weight_factor = snn_params["synapse_weight_factor"] # 
            self.noise_weight = snn_params["noise_weight"]
            self.fraction_stimulated = snn_params["fraction_stimulated"]  # Fraction of neurons to receive extra stimulation
            self.stimulation_probability = snn_params["stimulation_probability"]  # Chance of stimulation at any time step
            self.stimulator_synapse_weight = snn_params["stimulator_synapse_weight"]
            self.stimulation_time_resolution = snn_params["stimulation_time_resolution"] # 


            nest.ResetKernel()
            nest.SetKernelStatus({"resolution": snn_params["stimulation_time_resolution"], "print_time": True})

            