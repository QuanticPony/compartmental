# Copyright 2023 Unai Lería Fortea

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import compartmental as gcm
gcm.use_numpy()

import matplotlib.pyplot as plt
import numpy

sir_model = {
    "simulation": {
        "n_simulations": 100000,
        "n_executions": 10,
        "n_steps": 130
    },
    "compartiments": {
        "S": { 
            "initial_value": 1,
            "minus_compartiments": "I"
        },
        "I": { 
            "initial_value": "Io",
        },
        "R": { "initial_value": 0 },
    },
    "params": {
        "betta": {
            "min": 0.3,
            "max": 0.4
        },
        "mu": {
            "min": 0.01,
            "max": 0.1
        },
        "Io": {
            "min": 1e-6,
            "max": 1e-2
        }
    },
    "fixed_params": {
        "K_mean": 1
    },
    "reference": {
        "compartiments" : ["R"]
    },
    "results": {
        "save_percentage": 0.1
    }
}

SirModel = gcm.GenericModel(sir_model)

def evolve(m, *_):
    p_infected = m.betta * m.K_mean * m.I
    
    m.R += m.mu * m.I
    m.I += m.S * p_infected - m.I * m.mu
    m.S -= m.S * p_infected
    
SirModel.evolve = evolve



sample, sample_params = gcm.util.get_model_sample_trajectory(SirModel)

print(sample_params)
# print(sample)

# plt.plot(sample[:,0], color='green')
# plt.plot(sample[:,1], color='red')
# plt.plot(sample[:,2], color='black')
# plt.show()

# SirModel.populate_model_compartiments()
# SirModel.populate_model_parameters()

SirModel.run(sample[SirModel.compartiment_name_to_index["R"]], "sir.data")



results = gcm.util.load_parameters("sir.data")
weights = numpy.exp(-results[0]/numpy.min(results[0]))

percentiles = gcm.util.get_percentiles_from_results(SirModel, results, 30, 70)

plt.figure()
plt.fill_between(numpy.arange(percentiles.shape[2]), percentiles[0,0], percentiles[0,2], alpha=0.3)
plt.plot(sample[SirModel.compartiment_name_to_index["S"]])
plt.plot(sample[SirModel.compartiment_name_to_index["I"]])
plt.plot(sample[SirModel.compartiment_name_to_index["R"]])
plt.plot(numpy.arange(percentiles.shape[2]), percentiles[0,0])
plt.plot(numpy.arange(percentiles.shape[2]), percentiles[0,1])
plt.plot(numpy.arange(percentiles.shape[2]), percentiles[0,2])


fig, *axes = plt.subplots(1, len(results)-1)
for i, ax in enumerate(axes[0], 1):
    ax.hist(results[i], weights=weights)
    ax.vlines(sample_params[i-1], *ax.get_ylim())
    
plt.show()