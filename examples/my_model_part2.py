import compartmental 
compartmental.use_numpy()

import matplotlib.pyplot as plt
import numpy

model = {
    "simulation": {
        "n_simulations": 200000,
        "n_executions": 1,
        "n_steps": 100
    },
    "compartiments": {
        "Sh": { "initial_value": 0 },
        "S": { 
            "initial_value": 1,
            "minus_compartiments": "I"
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
            "min":0,
            "max":10
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
        "compartiments" : ["D"],
        "offset": "offset" 
    },
    "results": { 
        "save_percentage": 0.01
    }
}
MyModel = compartmental.GenericModel(model)
p_active = [1 if t<70 else 0 for t in range(model["simulation"]["n_steps"])]

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
    m.E[:]    = delta_S + (1-m.eta) * m.E
    
MyModel.evolve = evolve

sample, sample_params = compartmental.util.get_model_sample_trajectory(
    MyModel, p_active,
    **{"betta": 0.13,
        "Io": 1e-6,
        "phi": 0.1,
        "IFR": 0.01,
        "xi": 1/10}
)


results = compartmental.util.load_parameters("examples/my_model.data")
weights = numpy.exp(-results[0]/numpy.min(results[0]))

percentiles = compartmental.util.get_percentiles_from_results(MyModel, results, 30, 70, weights, p_active)#, weights)
try:
    # In case cupy is used
    percentiles = percentiles.get()
    sample = sample.get()
    weights = weights.get()
    results = results.get()
    sample_params = sample_params.get()
except AttributeError:
    pass

# Plot sample with a shadow of the results.
plt.figure()
plt.fill_between(numpy.arange(percentiles.shape[2]), percentiles[0,0], percentiles[0,2], alpha=0.3)
plt.plot(sample[MyModel.compartiment_name_to_index["D"]], 'black')
plt.plot(numpy.arange(percentiles.shape[2]), percentiles[0,1], '--', color='purple')
plt.plot(p_active, ':', color='green')

# Histograms with infered likelihood of the parameters
fig, *axes = plt.subplots(1, len(results)-1)
fig.set_figheight(4)
fig.set_figwidth(16)
for i, ax in enumerate(axes[0], 1):
    _5, _50, _95 = compartmental.util.weighted_quantile(results[i], [5, 50, 95], weights)
    ax.hist(results[i], weights=weights)
    ax.vlines(sample_params[i-1], *ax.get_ylim(), 'red')
    ax.vlines(_5, *ax.get_ylim(), 'green')
    ax.vlines(_50, *ax.get_ylim(), 'black')
    ax.vlines(_95, *ax.get_ylim(), 'purple')
    
plt.show()