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

import __future__
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as CNP

class ParametersManager:
    """Class used in model parameter population.
    """
    def __init__(self, configuration: dict[str, Any], model):
        self.configuration = configuration
        self.model = model
        
    def populate_params(self, **kargs):
        """Populates GenericModel.params array. If specific values are given in `kargs` those are used.

        Raises:
            Exception: If the model configuration could not fill all the array.
        """
        counter = 0
        dtype = []

        dtype = [[]]*len(self.model.param_to_index)
        N_SIMULATIONS = self.model.configuration["simulation"]["n_simulations"]

        for param_name, param_config in self.configuration["params"].items():
            param_index = self.model.param_to_index.get(param_name)
            if param_index is not None:
                TYPE = param_config.get("type", "float64")
                if TYPE == 'int':
                    dtype[param_index] = (param_name, CNP.int64)
                if TYPE == 'float64':
                    dtype[param_index] = (param_name, CNP.float64)


        # setattr(self.model, "params", CNP.zeros(
        #     (len(self.configuration["params"]), N_SIMULATIONS), dtype=dtype
        # ))
        setattr(self.model, "params", CNP.zeros(
            (N_SIMULATIONS), dtype=dtype
        ))

        for param_name, param_config in self.configuration["params"].items():
            counter += 1
            TYPE = param_config.get("type", "float64")

            if value := kargs.get(param_name, False):
                self.model.params[param_name] = value
                
                continue
            
            if TYPE == 'int':
                self.model.params[param_name] = CNP.random.randint(
                    param_config["min"], 
                    param_config["max"]+1, 
                    self.configuration["simulation"]["n_simulations"])
                
            if TYPE == 'float64':
                self.model.params[param_name] = CNP.random.random(
                    self.configuration["simulation"]["n_simulations"]) \
                        * (param_config["max"] - param_config["min"]) \
                        + param_config["min"]

        if counter < len(self.model.param_to_index.keys()):
            raise Exception("Parameters array could not be correctly created with current options.")


    def populate_fixed_params(self, **kargs):
        """Populates GenericModel.fixed_params with the configuration values.
        If specific values are given in `kargs` those are used.

        Raises:
            Exception: If the model configuration could not fill all the array.
        """
        setattr(self.model, "fixed_params", CNP.zeros(
            (len(self.configuration["fixed_params"]), 1), dtype=CNP.float64
        ))

        counter = 0
        for fparam_name, value in self.configuration["fixed_params"].items():
            fparam_index = self.model.fixed_param_to_index.get(fparam_name)
            if fparam_index is not None:
                counter += 1 
                self.model.fixed_params[fparam_index] = kargs.get(fparam_name, value)

        if counter < len(self.model.fixed_param_to_index.keys()):
            raise Exception("Fixed parameters array could not be correctly created with current options.")
