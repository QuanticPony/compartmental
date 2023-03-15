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
from io import TextIOWrapper

from typing import TYPE_CHECKING

from math import ceil
import copy

if TYPE_CHECKING:
    import numpy as CNP


def get_best_parameters(params, log_diff, save_percentage):
    "Retuns the best `save_percentage`% `params` of the simulations given their `log_diff` with real data." 
    save_count: int = ceil(log_diff.size*save_percentage*0.01)
    saved_params = CNP.zeros((save_count, params.shape[0]), dtype=CNP.float64)
    saved_log_diff = CNP.zeros((save_count, 1), dtype=CNP.float64)

    log_diff_index_sorted = CNP.argpartition(log_diff, save_count, 0)[0:save_count]
    
    saved_params[:,:] = CNP.take(params, log_diff_index_sorted, 1).T
    saved_log_diff[:] = CNP.take(log_diff, log_diff_index_sorted)
    return saved_params, saved_log_diff


def progress_bar(prefix, progress, total, *, sufix="", end='\r', len=10):
    """Prints a progress bar on standar output.

    Args:
        prefix (str): Prefix to the progress bar.
        progress (int|float): Progress value.
        total (int|float): Total progess posible.
        sufix (str, optional): Sufix to the progess bar. Defaults to "".
        end (str, optional): End value, set to `\\n` at the end. Defaults to '\r'.
        len (int, optional): Length of progress bar. Defaults to 10.
    """
    per = len * progress/float(total)
    print(f"\r{prefix} -> ||{'▮'*int(per) + '▯'*(len-int(per))} ||{per*100/len:.2f}%  {sufix}", end=end)


def save_parameters_no_diff(file: str, params_names: list[str], params: list[list[float]], *, execution_number=0):
    """Saves the parameters with the given names without the diff column.

    Args:
        file (str): Filename or path to file.
        params_names (list[str]): Name of parameters.
        params (list[list[float]]): Parameters array.
        execution_number (int, optional): Number of the execution. If `0` the header is printed. Defaults to 0.
    """
    with open(file, 'a' if execution_number!=0 else 'w') as file_out:
        CNP.savetxt(file_out, params.T, delimiter=',', comments='', header=",".join(params_names) if execution_number==0 else "")


def save_parameters(file: str, params_names: list[str], params: list[list[float]], log_diff: list[float], *, execution_number=0):
    """Saves the parameters with the given names including the diff column.

    Args:
        file (str): Filename or path to file.
        params_names (list[str]): Name of parameters.
        params (list[list[float]]): Parameters array.
        log_diff (list[float]): Diff array.
        execution_number (int, optional): Number of the execution. If `0` the header is printed. Defaults to 0.
    """
    with open(file, 'a' if execution_number!=0 else 'w') as file_out:
        CNP.savetxt(file_out, CNP.concatenate((log_diff, params), 1) , delimiter=',', comments='', header=",".join(["log_diff", *params_names]) if execution_number==0 else "")

def load_parameters(file: str):
    """Loads parameters from file with the same format as `save_parameters` and `save_parameters_no_diff`.

    Args:
        file (str): Filename or path to file.

    Returns:
        (list[list[float]]): Parameters array. First index selects the column (parameter).
    """
    with open(file, 'r') as file_in:
        results = CNP.loadtxt(file_in, delimiter=',', skiprows=1).T
    return results


def get_model_sample_trajectory(model, *args, **kargs):
    """Executes the model with `n_simulations = 1` and `n_executions = 1`.
    Returns all the intermediate states and the parameters.

    Args:
        model (GenericModel): Model to execute.

    Returns:
        (list[list[float]], list[list[float]]): Tuple of all states history and corresponding params.
    """
    prev_config = copy.deepcopy(model.configuration)

    model.configuration["simulation"]["n_simulations"] = 1
    model.configuration["simulation"]["n_executions"] = 1
    
    model.populate_model_parameters(*args, **kargs)
    model.populate_model_compartiments(*args, **kargs)
    saved_state = CNP.zeros((model.configuration["simulation"]["n_steps"], model.state.shape[0]))
    for step in range(model.configuration["simulation"]["n_steps"]):
        model.evolve(model, step, *args, **kargs)
        saved_state[step] = model.state[:, 0]
        
    model.configuration.update(prev_config)
    return saved_state.T, model.params[:, 0]


def get_percentiles_from_results(model, results, p_minor=5, p_max=95, *args, **kargs):
    """Returns an array of percentils `p_minor=5`, median and `p_max=95` of the given model and results.

    Args:
        model (GenericModel): Model used to generate the `results`.
        results (list[list[float]]): Result parameters of `model` execution.
        p_minor (int, optional): Smaller percentile. Defaults to 5.
        p_max (int, optional): Bigger percentile. Defaults to 95.

    Returns:
        (list[int, int, list[float]]): First index represents the reference defined in `reference.compartiments`. \
            Second index represents  `p_minor`, median or `p_max=`. Final represents the step in the simulation.
    """
    reference_mask = CNP.array([model.compartiment_name_to_index[c] for c in model.configuration["reference"]["compartiments"]])
    
    results_no_diff = results[1:]
    results_percentiles = CNP.zeros((reference_mask.shape[0], 3, model.configuration["simulation"]["n_steps"]))
    
    prev_config = copy.deepcopy(model.configuration)

    model.configuration["simulation"]["n_simulations"] = results.shape[1]
    model.configuration["simulation"]["n_executions"] = 1
    
    model.populate_model_parameters(*args, **kargs)
    model.params[:] = results_no_diff[:]
    model.populate_model_compartiments(*args, **kargs)
    
    def inner(model, step, reference, reference_mask, *args, **kargs):
        model.evolve(model, step, *args, **kargs)
        aux = CNP.take(model.state, reference_mask, 0)
        
        results_percentiles[:, 0, step] += CNP.percentile(aux, p_minor, 1)
        results_percentiles[:, 1, step] += CNP.median(aux, 1)
        results_percentiles[:, 2, step] += CNP.percentile(aux, p_max, 1)
        
    def outer(model, *args, **kargs):
        ...
        
    model._internal_run_(
        inner, (reference_mask,), 
        outer, (), 
        None, None,
        *args, **kargs
    )
    model.configuration.update(prev_config)
    
    return results_percentiles
