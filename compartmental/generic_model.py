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
 
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as CNP


from .util import *
from .parameters import ParametersManager

import copy

class GenericModel:
    """Creates a compartmental model from a dictionary and setting an `evolve` method.
    """

    def get_all_params_names(self):
        """Returns a set of all parameters names, fixed or not.

        Returns:
            (set[str]): Set of all parameters names.
        """
        return set(list(self.param_to_index.keys()) + list(self.fixed_param_to_index.keys()))


    def __init__(self, configuration: dict[str, Any]):
        """Creates a model from the configuration given.

        Args:
            configuration (dict[str, Any]): Model configuration.
        """
        self.configuration: dict[str, Any] = copy.deepcopy(configuration)
        
        self.param_to_index: dict[str, int] = { k:i for i,k in enumerate(self.configuration["params"].keys()) }
        self.fixed_param_to_index: dict[str, int] = { k:i for i,k in enumerate(self.configuration["fixed_params"].keys()) }
        self.compartment_name_to_index: dict[str, int] = { k:i for i,k in enumerate(self.configuration["compartments"].keys()) }


    def populate_model_parameters(self, **kargs):
        """Populates params array. Assigns shortcuts to call them by their name as an attribute.
        """
        parameter_manager = ParametersManager(self.configuration, self)
        
        REFERENCE_OFFSET = self.configuration["reference"].get("offset", 0)
        # Set offset value if it is a str reference
        if isinstance(REFERENCE_OFFSET, str):
            self.configuration["params"][REFERENCE_OFFSET].update({"type":"int"})

        parameter_manager.populate_params(**kargs)
        parameter_manager.populate_fixed_params()
        
        for param in self.configuration["params"].keys():
            setattr(self, param, self.params[param])
            
        for fparam in self.configuration["fixed_params"].keys():
            setattr(self, fparam, self.fixed_params[self.fixed_param_to_index[fparam]])
            
        if isinstance(REFERENCE_OFFSET, str):
            self.reference_offset = self.params[REFERENCE_OFFSET]
        else:
            self.reference_offset = 0
            

        
        
    def populate_model_compartments(self, **kargs):
        """Populates compartments array. Assigns shortcuts to call them by their name as an attribute.
        """
        N_SIMULATIONS = self.configuration["simulation"]["n_simulations"]
        self.state = CNP.zeros(
            (len(self.configuration["compartments"]), N_SIMULATIONS), dtype=CNP.float64
        )
        self.log_diff = CNP.zeros((N_SIMULATIONS, 1), dtype=CNP.float64)
        
        for c,i in self.compartment_name_to_index.items():
            C = self.configuration["compartments"][c]
            initial_value = C["initial_value"]
            if isinstance(initial_value, str):
                if initial_value in self.param_to_index.keys():
                    self.state[i,:] = self.params[initial_value]
                continue
            self.state[i,:] = initial_value
           
        for c,i in self.compartment_name_to_index.items():
            C = self.configuration["compartments"][c]
            minus = C.get("minus_compartments", False)
            if not minus:
                continue
            if not isinstance(minus, list):
                minus = [minus]
            for m in minus:
                self.state[i,:] -= self.state[self.compartment_name_to_index[m],:]
                
        for comp in self.configuration["compartments"].keys():
            setattr(self, comp, self.state[self.compartment_name_to_index[comp]])
                

    def evolve(self, step, *args, **kargs):
        """This method must be overwritten to complete the model initialization.

        Args:
            step (int): Step of simulation. Simulation ends when `step = simulation.n_steps`
        """
        ...


    def get_diff(self, step, reference, reference_mask):
        """Returns a value that represents the distance from the simulation to the reference.
        This function can be overwritten if other deffinition os distance is needed.
        
        Args:
            step (int): Step of simulation. Simulation ends when `step = simulation.n_steps`
            reference (list[list[float]]): Reference(s) values.
            reference_mask (list[int]): Mask to obtain simulation values to compare with the reference(s). 

        Returns:
            (list[float]): Distance from simulations to reference(s).
        """
        index = step + self.reference_offset
        # To only take the diff on the same range for all simulations
        diff = CNP.absolute(CNP.take(self.state, reference_mask, 0)[0].T-reference[CNP.clip(index, 0, self.N_STEPS-1)]) * \
               ((self.reference_offset.max()<=index) * (index<=self.N_STEPS))
               
        return CNP.log(diff + 1)


    def _internal_run_(self, inner, inner_args: list, outer, outer_args:list,  reference, save_file:str, *args, exclude_pupulate:bool=False, **kargs):
        """Internal function that executes the model.

        Args:
            inner (function): Function to call in the main loop.
            inner_args (list): Args given to `inner`.
            outer (function): Function to call after the main loop.
            outer_args (list): Args given to `outer`.
            reference (list[list[float]]): Reference values used to compare with the simulation.
            save_file (str): Filename of path to file.
            exclude_populate (bool, optional): If `False` params and compartments are populated with random values. Defaults to False.
        """
        N_EXECUTIONS = self.configuration["simulation"]["n_executions"]
        self.N_STEPS = self.configuration["simulation"]["n_steps"]

        for execution in range(N_EXECUTIONS):
            progress_bar(f"Model running: ", execution, N_EXECUTIONS, len=min(20, max(N_EXECUTIONS,5)))
            
            if not exclude_pupulate:
                self.populate_model_parameters(**kargs)
                self.populate_model_compartments(**kargs)

            self._min_offset_: int = self.reference_offset.min()
            self.log_diff[:] = 0

            # for step in range(self.N_STEPS):
            step = CNP.int64(0)
            while (self.reference_offset + step < self.N_STEPS).any():
                inner(self, step, reference, *inner_args, **kargs)
                step += 1
            outer(self, *outer_args, execution_number=execution, **kargs)
            
        progress_bar(f"Model running: ", N_EXECUTIONS, N_EXECUTIONS, len=min(20, max(N_EXECUTIONS,5)), end='\n')


    def run_no_diff(self, save_file: str, *args, **kargs):
        """Runs the model without computing the diference any reference(s).

        Args:
            save_file (str): Filename of path to file.
        """
        self._internal_run_(
            self.evolve, args, 
            save_parameters_no_diff, (save_file, self.param_to_index.keys(),  self.params), 
            None, save_file,
            *args, **kargs
        )


    def run(self, reference, save_file: str, *args, **kargs):
        """Runs the model computing the diference from the reference(s).

        Args:
            reference (list[list[float]]): Reference(s) values.
            save_file (str): Filename of path to file.
        """
        
        reference_mask = CNP.array([self.compartment_name_to_index[c] for c in self.configuration["reference"]["compartments"]])
        
        def inner(model, step, reference, reference_mask, *args, **kargs):
            model.evolve(model, step, *args, **kargs)
            self.log_diff[:,0] += model.get_diff(step, reference, reference_mask)
        
        def outer(model, save_file, *args, execution_number, **kargs):
            best_params, best_log_diff = get_best_parameters(model.params, model.log_diff, model.configuration["results"]["save_percentage"])
            save_parameters(save_file, model.param_to_index.keys(), best_params, best_log_diff, execution_number=execution_number)
            
        self._internal_run_(
            inner, (reference_mask, *args), 
            outer, (save_file,), 
            reference, save_file, 
            *args, **kargs
        )
