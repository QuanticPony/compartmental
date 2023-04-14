<!-- Copyright 2023 Unai Lería Fortea

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

For a more in-depth example check the Jupiter notebook with the same name, or check out https://github.com/QuanticPony/compartmental/blob/master/examples/sir.ipynb


$$
\begin{align}
    \nonumber \dot{S} &= -\beta \langle k \rangle \frac{I}{N}S. \\
    \nonumber \dot{I} &= \beta \langle k \rangle \frac{I}{N}S - \mu I. \\
    \nonumber \dot{R} &= \mu I.
\end{align}
$$

```json
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
            "max": 0.4
        },
        "mu": {
            "min": 0.01,
            "max": 0.2
        },
        "Io": {
            "min": 1e-6,
            "max": 1e-4
        }
    },
    "fixed_params": {
        "K_mean": 1
    },
    "reference": {
        "compartments" : ["R"]
    },
    "results": {
        "save_percentage": 0.1
    }
}
```


Now we need to define the evolution function of the system and assign it to the model:
```py
import compartmental as gcm
gcm.use_numpy()


SirModel = gcm.GenericModel(sir_model)

def evolve(m, *args, **kargs):
    p_infected = m.betta * m.K_mean * m.I
    
    m.R += m.mu * m.I
    m.I += m.S * p_infected - m.I * m.mu
    m.S -= m.S * p_infected
    
SirModel.evolve = evolve
```


Once the model is defined and the evolution function is set we can create a trajectory of the model. We can set specific values for the random parameters as follows:

```py
sample, sample_params = gcm.util.get_model_sample_trajectory(SirModel, **{"betta":0.2, "mu":0.08, "Io": 1e-5})
```

Now we apply the automatic adjustment of the model. Keep in mind it will only work if the initial ranges of the `params` are set close to the optimal values.
```py
for i in range(7):
    SirModel.run(sample[SirModel.compartment_name_to_index["R"]], f"sir_temp{i}.data")
    
    results = gcm.util.load_parameters(f"sir_temp{i}.data")
    
    gcm.util.auto_adjust_model_params(SirModel, results)
```

Finally we run the model once again to get the final photo:
```py
SirModel.run(sample[SirModel.compartment_name_to_index["R"]], "sir.data")
results = gcm.util.load_parameters("sir.data")
```

<table>
    <tr> 
        <td> 
            <h3 align='center'> Not adjusted
        </td> 
        <td> <img src="../../images/sir_1.png"  alt="1" width = 500px height = 640px> </td>
        <td> <img src="../../images/sir_2.png" alt="2" width = 500px height = 640px> </td>
    </tr> 
    <tr>
        <td> 
            <h3 align='center'> With automatic adjustment
        </td> 
        <td><img src="../../images/sir_3.png" alt="3" width = 500px height = 640px></td>
        <td><img src="../../images/sir_4.png" alt="4" width = 500px height = 640px>
        </td>
    </tr>
</table>




Code used for the plots:

```py
weights = numpy.exp(-results[0]/numpy.min(results[0]))

percentiles = gcm.util.get_percentiles_from_results(SirModel, results, 30, 70)
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
plt.plot(sample[SirModel.compartment_name_to_index["R"]], 'brown')
plt.plot(numpy.arange(percentiles.shape[2]), percentiles[0,1], '--', color='purple')


fig, *axes = plt.subplots(1, len(results)-1)
for i, ax in enumerate(axes[0], 1):
    ax.hist(results[i], weights=weights)
    ax.vlines(sample_params[i-1], *ax.get_ylim(), 'red')
    
plt.show()
```