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

import compartmental 
compartmental.use_numpy()

import matplotlib.pyplot as plt
import numpy
model = {
    "simulation": {
        "n_simulations": 100000,
        "n_executions": 1,
        "n_steps": 100
    },
    "compartments": {
        "Sh": { "initial_value": 0 },
        "S": { 
            "initial_value": 1,
            "minus_compartments": "I"
        },
        "E": { "initial_value": 0 },
        "I": { 
            "initial_value": "Io",
        },
        "R": { "initial_value": 0 },
        "Pd": { "initial_value": 0 },
        "D": { "initial_value": 0 },
    },
    "params": {
        "betta": {
            "min": 0.01,
            "max": 0.3
        },
        "Io": {
            "min": 1e-8,
            "max": 1e-5
        },
        "phi": {
            "min": 0,
            "max": 1
        },
        "IFR": {
            "min":0.006,
            "max":0.014
        },
        "xi": {
            "min":1/16,
            "max":1/6
        },
        "offset": {
            "min":4,
            "max":12
        }
    },
    "fixed_params": {
        "K_active": 12.4,
        "K_lockdown": 2.4,
        "sigma": 3.4,
        "mu": 1/4.2,
        "eta":1/5.2
    },
    "reference": {
        "compartments" : ["D"],
        "offset": "offset" 
    },
    "results": { 
        "save_percentage": 1
    }
}

MyModel = compartmental.GenericModel(model)

def evolve(m, time, p_active, *args, **kargs):
    ST = m.S + m.Sh
    sh = (1 - m.I) ** (m.sigma - 1)

    P_infection_active = 1- (1- m.betta * m.I) ** m.K_active
    P_infection_lockdown = 1- (1- m.betta * m.I) ** m.K_lockdown

    P_infection = p_active[time] * P_infection_active + (1-p_active[time]) * (1-sh*(1-m.phi)) * P_infection_lockdown


    m.Sh[:]    = ST * (1-p_active[time])*sh*(1-m.phi)
    delta_S = ST * P_infection
    m.S[:]     = (ST - m.Sh)  - delta_S
   
    m.D[:]     = m.xi * m.Pd
    m.R[:]     = m.mu * (1-m.IFR)  * m.I + m.R
    m.Pd[:]    = m.mu * m.IFR  * m.I + (1-m.xi) * m.Pd
    m.I[:]     = m.eta  * m.E + (1- m.mu) * m.I
    m.E[:]     = delta_S + (1-m.eta) * m.E
    
MyModel.evolve = evolve

p_active = [1 if t<70 else 0.1 for t in range(model["simulation"]["n_steps"])]

sample, sample_params = compartmental.util.get_model_sample_trajectory(
    MyModel, p_active,
    **{"betta": 0.13,
        "Io": 1e-6,
        "phi": 0.1,
        "IFR": 0.01,
        "xi": 1/10,
        "offset": 8}
)


ITERS = 2
# This array is created to store min and max of params configuration in order to see the adjustment in action.
saved_params_lims = numpy.zeros((len(MyModel.configuration["params"]), 2, ITERS))

# Main loop of adjustments:
# 1. Run
# 2. Read results
# 3. Compute weights
# 4. Adjuts configuration
for i in range(ITERS):
    MyModel.run(sample[MyModel.compartment_name_to_index["D"]], f"my_model{i}.data", p_active)
    
    results = compartmental.util.load_parameters(f"my_model{i}.data")
    
    compartmental.util.auto_adjust_model_params(MyModel, results)
    
    # Needed to see the max and min evolution in the adjustment
    for p, v in MyModel.configuration["params"].items():
        saved_params_lims[MyModel.param_to_index[p], 0, i] = v["min"]
        saved_params_lims[MyModel.param_to_index[p], 1, i] = v["max"]