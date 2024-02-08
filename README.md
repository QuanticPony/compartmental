<!-- Copyright 2023 Unai Lería Fortea & Pablo Vizcaíno García

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

[![PyPI Downloads](https://img.shields.io/pypi/dm/compartmental.svg?label=downloads)](https://pypi.org/project/compartmental/)
[![PyPI Version](https://img.shields.io/pypi/v/compartmental?)](https://pypi.org/project/compartmental/)

![Commit activity](https://img.shields.io/github/commit-activity/m/QuanticPony/compartmental)
[![License](https://img.shields.io/pypi/l/compartmental)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/QuanticPony/compartmental/ci-master.yml)](https://github.com/QuanticPony/compartmental/actions)

[![Python Version](https://img.shields.io/pypi/pyversions/compartmental)](https://pypi.org/project/compartmental/)
[![Wheel](https://img.shields.io/pypi/wheel/compartmental)](https://pypi.org/project/compartmental/)

<br></br>
<h1 align="center">
compartmental
</h1>
<h2 align="center">
Utility tools for Approximate Bayesian computation on compartmental models 
</h2>

<br>
<div align="center">

<a href="https://quanticpony.github.io/compartmental/">
<img src=https://img.shields.io/github/deployments/QuanticPony/compartmental/github-pages?label=documentation>
</a>
<br></br></br>
<h3 align="center">

</h3>

</div>


# SIR Example
(Example taken from [examples](https://quanticpony.github.io/compartmental/examples/SIR/))

To make a SIR model you will need a configuration and an evolution function:
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

```python
import compartmental as gcm
gcm.use_cupy() // or numpy in case you don't have access to a gpu.


SirModel = gcm.GenericModel(sir_model)

def evolve(m, *args, **kargs):
    p_infected = m.betta * m.K_mean * m.I

    m.R += m.mu * m.I
    m.I += m.S * p_infected - m.I * m.mu
    m.S -= m.S * p_infected

SirModel.evolve = evolve
```

That's it! 

You can now execute it on your GPU and fit the model to some data. [Have a look at the example!](https://quanticpony.github.io/compartmental/examples/SIR/)

![image](https://github.com/QuanticPony/compartmental/assets/67756626/fdd7147c-a0c1-48c1-bac8-335257f1c3ee)


Or have a look into a more elavorated model [here](https://quanticpony.github.io/compartmental/examples/MY_MODEL/).
![image](https://github.com/QuanticPony/compartmental/assets/67756626/613cf6fd-e38e-428f-a088-dd7b822bf54c)


# Instalation
**compartmental** releases are available as wheel packages on [PyPI](https://pypi.org/project/compartmental/). You can install the last version using `pip`:
```
pip install compartmental
```


# Documentation
Documentations is automatically generated from code on main push and hosted in github-pages [here](https://quanticpony.github.io/compartmental/).

# Help
Just open an issue with the `question` tag ([or clic here](https://github.com/QuanticPony/compartmental/issues/new?assignees=QuanticPony&labels=question&template=question.md&title=)), I would love to help!

# Contributing
You can contribute with:

* Examples
* Documentation
* [Bug report/fix](https://github.com/QuanticPony/compartmental/issues/new?assignees=QuanticPony&labels=bug&template=bug_report.md&title=)
* [Features](https://github.com/QuanticPony/compartmental/issues/new?assignees=QuanticPony&labels=new-feature&template=feature_request.md&title=)
* Code

Even only feedback is greatly apreciated. 

Just create an issue and let me know you want to help! 


# Licensing
**compartmental** is released under the **Apache License Version 2.0**.
