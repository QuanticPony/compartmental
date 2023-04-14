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
        "n_executions": 1,
        "n_steps": 130
    },
    "compartments": {
        "S": { 
            "initial_value": 1,
            "minus_compartments": "I"
        },
        "I": { 
            "initial_value": "Io",
        },
        "R": { "initial_value": 0 },
    },
    "params": {
        "betta": {
            "min": 0.1,
            "max": 0.4,
            "min_limit": 0.1,
            "max_limit": 0.4
        },
        "Io": {
            "min": 1e-6,
            "max": 1e-4,
            "min_limit": 1e-7,
            "max_limit": 1e-2
        },
        "To": {
            "type": "int",
            "min": 0,
            "min_limit": 0,
            "max": 8
        }
    },
    "fixed_params": {
        "K_mean": 1,
        "mu": 0.08
    },
    "reference": {
        "compartments" : ["R"],
        "offset": "To"
    },
    "results": {
        "save_percentage": 0.1
    }
}

SirModel = gcm.GenericModel(sir_model)

def evolve(m, *args, **kargs):
    p_infected = m.betta * m.K_mean * m.I
    
    m.R += m.mu * m.I
    m.I += m.S * p_infected - m.I * m.mu
    m.S -= m.S * p_infected
    
SirModel.evolve = evolve

OFFSET = 3

sample, sample_params = gcm.util.get_model_sample_trajectory(SirModel, **{"betta":0.2, "Io": 1e-5, "To": OFFSET})

reference = numpy.copy(sample[SirModel.compartment_name_to_index["R"]])
gcm.util.offset_array(reference, OFFSET)


ITERS = 7
# This array is created to store min and max of params configuration in order to see the adjustment in action.
saved_params_lims = numpy.zeros((len(SirModel.configuration["params"]), 2, ITERS))

# Main loop of adjustments:
# 1. Run
# 2. Read results
# 3. Compute weights
# 4. Adjuts configuration
for i in range(ITERS):
    SirModel.run(reference, "sir_temp.data")
    
    results = gcm.util.load_parameters("sir_temp.data")
    weights = numpy.exp(-results[0]/numpy.min(results[0]))
    
    gcm.util.auto_adjust_model_params(SirModel, results, weights)
    
    # Needed to see the max and min evolution in the adjustment
    for p, v in SirModel.configuration["params"].items():
        saved_params_lims[SirModel.param_to_index[p], 0, i] = v["min"]
        saved_params_lims[SirModel.param_to_index[p], 1, i] = v["max"]

# Plot evolution of the parameters adjustment
for i, (k,v) in enumerate(SirModel.configuration["params"].items()):
    plt.figure()
    plt.title(k)
    plt.fill_between(range(ITERS), saved_params_lims[i, 0, :], saved_params_lims[i, 1, :])
    
# Update for final photo with 3M samples
SirModel.configuration.update({
    "simulation": {
        "n_simulations": 100000,
        "n_executions": 10,
        "n_steps": 130
    },
    "results": {
        "save_percentage": 0.01
    }
})

SirModel.run(reference, "sir.data")

results = gcm.util.load_parameters("sir.data")
weights = numpy.exp(-results[0]/numpy.min(results[0]))

percentiles = gcm.util.get_percentiles_from_results(SirModel, results, 30, 70, weights)
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
plt.plot(sample[SirModel.compartment_name_to_index["S"]], 'green')
plt.plot(sample[SirModel.compartment_name_to_index["I"]], 'orange')
plt.plot(sample[SirModel.compartment_name_to_index["R"]], 'black')
plt.plot(reference, 'brown')
plt.plot(numpy.arange(percentiles.shape[2]), percentiles[0,1], '--', color='purple')


fig, *axes = plt.subplots(1, len(results)-1)
for i, ax in enumerate(axes[0], 1):
    _5, _50, _95 = gcm.util.weighted_quantile(results[i], [5, 50, 95], weights)
    ax.hist(results[i], weights=weights)
    ax.vlines(sample_params[i-1], *ax.get_ylim(), 'red')
    ax.vlines(_5, *ax.get_ylim(), 'green')
    ax.vlines(_50, *ax.get_ylim(), 'black')
    ax.vlines(_95, *ax.get_ylim(), 'purple')
    
plt.show()


values = gcm.util.get_trajecty_selector(SirModel, results, weights, reference)
print(values)