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

seir_model = {
    "simulation": {
        "n_simulations": 100000,
        "n_executions": 1,
        "n_steps": 230
    },
    "compartiments": {
        "S": { 
            "initial_value": 1,
            "minus_compartiments": "I"
        },
        "E": { "initial_value": 0 },
        "I": { 
            "initial_value": "Io",
        },
        "R": { "initial_value": 0 },
    },
    "params": {
        "betta": {
            "min": 0.1,
            "max": 0.3
        },
        "Io": {
            "min": 1e-6,
            "max": 1e-4
        }
    },
    "fixed_params": {
        "K_mean": 1,
        "mu": 0.07,
        "eta":0.08
    },
    "reference": {
        "compartiments" : ["R"]
    },
    "results": {
        "save_percentage": 0.1
    }
}

SeirModel = gcm.GenericModel(seir_model)

def evolve(m, *args, **kargs):
    p_infected = m.betta * m.K_mean * m.I
    
    m.R += m.mu * m.I
    m.I += m.E * m.eta - m.I * m.mu
    m.E += m.S * p_infected - m.E * m.eta
    m.S -= m.S * p_infected
    
SeirModel.evolve = evolve

sample, sample_params = gcm.util.get_model_sample_trajectory(SeirModel, **{"betta": 0.2, "Io":6e-5})


SeirModel.run(sample[SeirModel.compartiment_name_to_index["R"]], "seir.data")

results = gcm.util.load_parameters("seir.data")
weights = numpy.exp(-results[0]/numpy.min(results[0]))

percentiles = gcm.util.get_percentiles_from_results(SeirModel, results, 30, 70)
try:
    # In case cupy is used
    percentiles = percentiles.get()
    sample = sample.get()
    weights = weights.get()
    results = results.get()
    sample_params = sample_params.get()
except AttributeError:
    pass

plt.figure()
plt.fill_between(numpy.arange(percentiles.shape[2]), percentiles[0,0], percentiles[0,2], alpha=0.3)
plt.plot(sample[SeirModel.compartiment_name_to_index["S"]], 'green')
plt.plot(sample[SeirModel.compartiment_name_to_index["E"]], 'red')
plt.plot(sample[SeirModel.compartiment_name_to_index["I"]], 'orange')
plt.plot(sample[SeirModel.compartiment_name_to_index["R"]], 'brown')
plt.plot(numpy.arange(percentiles.shape[2]), percentiles[0,1], '--', color='purple')


fig, *axes = plt.subplots(1, len(results)-1)
for i, ax in enumerate(axes[0], 1):
    ax.hist(results[i], weights=weights)
    ax.vlines(sample_params[i-1], *ax.get_ylim(), 'red')
    
plt.show()