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
import matplotlib.pyplot as plt

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
    
    saved_params[:,:] = CNP.take(params, log_diff_index_sorted, 1)[:,:,0].T
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
        import numpy as np
        if np != CNP:
            _params = CNP.asnumpy(params)
        else:
            _params = params
        np.savetxt(file_out, params.T, delimiter=',', comments='', header=",".join(params_names) if execution_number==0 else "")


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
        import numpy as np
        if np != CNP:
            np.savetxt(file_out, CNP.asnumpy(CNP.concatenate((log_diff, params), 1)) , delimiter=',', comments='', header=",".join(["log_diff", *params_names]) if execution_number==0 else "")
        else:
            _log_diff = log_diff
            _params = params
            np.savetxt(file_out, np.concatenate((_log_diff, _params), 1) , delimiter=',', comments='', header=",".join(["log_diff", *params_names]) if execution_number==0 else "")

        
def load_parameters(file: str):
    """Loads parameters from file with the same format as `save_parameters` and `save_parameters_no_diff`.

    Args:
        file (str): Filename or path to file.

    Returns:
        (list[list[float]]): Parameters array. First index selects the column (parameter).
    """
    with open(file, 'r') as file_in:
        import numpy as np
        results = np.loadtxt(file_in, delimiter=',', skiprows=1).T
    return CNP.asarray(results)


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


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """From: https://stackoverflow.com/a/29677616
    
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 100]!
    
    Args:
        values (list[float]): Array with data.
        quantiles (list[float]): Array with many quantiles needed.
        sample_weight (list[float], optional): Array of the same length as `array`. Defaults to None.
        values_sorted (bool): If True, then will avoid sorting of initial array.
        old_style (bool): If True, will correct output to be consistent with numpy.percentile.
        
    Returns:
        (list[float]): Array with computed quantiles.
    """
    quantiles = CNP.array(quantiles, dtype=CNP.float64)
    quantiles /= 100.
    
    if sample_weight is None:
        sample_weight = CNP.ones(len(values))
    sample_weight = CNP.array(sample_weight)

    if not values_sorted:
        sorter = CNP.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = CNP.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= CNP.sum(sample_weight)
    return CNP.interp(quantiles, weighted_quantiles, values)


def get_percentiles_from_results(model, results, p_minor=5, p_max=95, weights=None, *args, **kargs):
    """Returns an array of percentils `p_minor=5`, median and `p_max=95` of the given model and results.

    Args:
        model (GenericModel): Model used to generate the `results`.
        results (list[list[float]]): Result parameters of `model` execution.
        p_minor (int, optional): Smaller percentile. Defaults to 5.
        p_max (int, optional): Bigger percentile. Defaults to 95.
        weights (list[float]|None): Results weights. Defaults to None.

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
        aux_sorted = CNP.sort(aux)
        
        if weights is not None:
            percentile = lambda x,p: weighted_quantile(x[0], p, weights, True, False)
        else:
            percentile = lambda x,p: CNP.percentile(x, p)

        results_percentiles[:, 0, step] += percentile(aux_sorted[0], p_minor)
        results_percentiles[:, 1, step] += percentile(aux_sorted[0], 50)
        results_percentiles[:, 2, step] += percentile(aux_sorted[0], p_max)

        
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


def auto_adjust_model_params(model, results, weights=None, params=None):
    """Adjusts limits of model params. If `params` is specified only those are adjusted.

    Args:
        model (GenericModel): Model to optimize.
        results (list[list[float]]): Results from running the model.
        weights (list[float], optional): Results weights. Defaults to None.
        params (list[str], optional): Names of params to optimice. Defaults to None.
    """
    if weights is not None:
        percentile = lambda x,p: weighted_quantile(x, p, weights, True, False)
    else:
        percentile = lambda x,p: CNP.percentile(x, p)
         
    for c, i in model.param_to_index.items():
        if isinstance(params, list):
            if isinstance(params[0], str):
                if c not in params:
                    continue
            
        aux = CNP.sort(results[i+1])
        _5 = percentile(aux, 5)
        _50 = percentile(aux, 50)
        _95 =percentile(aux, 95)
        M:dict = model.configuration["params"][c]
        
        distm = _50 - _5
        distM = _95 - _50
        # dist = distm + distM
        
        model.configuration["params"][c].update({
            # "min" : CNP.clip(_50 - dist*(distm/distM), M.get("min_limit", None), M.get("max_limit", None)), # min(_5, (M["min"]+_5)/2),
            # "max" : CNP.clip(_50 + dist*(distM/distm), M.get("min_limit", None), M.get("max_limit", None)) # max(_95, (M["max"]+_95)/2)
            "min" : CNP.clip(min(_50 - distm*(distm/distM), (M["min"]+_5)/2), M.get("min_limit", M["min"]), M.get("max_limit", M["max"])),
            "max" : CNP.clip(max(_50 + distM*(distM/distm), (M["max"]+_95)/2), M.get("min_limit", M["min"]), M.get("max_limit", M["max"]))
        
        })
        

def get_trajecty_selector(model, results, weights, reference=None):
    """Creates an interactive plot and histograms of results. When a histogram is clicked the value of
    that parameter changes to the selected value.

    Args:
        model (GenericModel): Model used for the trajectory.
        results (list[list[float]]): Results from running the model.
        weights (list[float], optional): Results weights. Defaults to None.
        reference (list[list[float]], optional): If give, is printed to the trajectory. Defaults to None.

    Returns:
        (dict[str, float]): Dictionary with the manually selected params.
    """
    prev_config = copy.deepcopy(model.configuration)
    fig_sample, ax_sample = plt.subplots()
    
    _range = CNP.arange(model.configuration["simulation"]["n_steps"])
    
    # Params used for the trajectory are saved here. This is returned
    values = {p:v["min"] for p,v in model.configuration["params"].items()}
    
    fig, *axes = plt.subplots(1, len(results)-1)
    for (p, i), ax in zip(model.param_to_index.items(), axes[0]):
        _5, _50, _95 = weighted_quantile(results[i+1], [5, 50, 95], weights)
        ax.set_xlabel(p)
        ax.hist(results[i+1], weights=weights)
        xlim = ax.get_xlim()
        ax.vlines(_5, *ax.get_ylim(), 'green')
        ax.vlines(_50, *ax.get_ylim(), 'black')
        ax.vlines(_95, *ax.get_ylim(), 'purple')
        line, _ = ax.plot([(_5+_50)/2,(_5+_50)/2 ],  ax.get_ylim(), 'red', ':')
        
        # Define a picker por the param ax
        def picker_builder(param, vline):
            def picker(self, event):
                x = event.xdata
                # Update values and vline xdata
                values.update({param:x})
                vline.set_xdata([x,x])
                # Update trajectory
                for data, line in zip(update(), sample_lines):
                    line.set_ydata(data)
                
                fig_sample.canvas.draw_idle()
                fig.canvas.draw_idle()
                return True, {}
            return picker
            
        line.set_picker(picker_builder(p, line))
        values.update({p:(_5+_50)/2})
        ax.set_xlim(xlim)

    def update():
        sample, _ = get_model_sample_trajectory(model, **values)
        return sample

    sample = update()
    list_of_sample_lines = []
    for s in sample:
        list_of_sample_lines.append(_range)
        list_of_sample_lines.append(s)
        list_of_sample_lines.append('-')
        
    sample_lines = ax_sample.plot(*list_of_sample_lines)
    if reference is not None:
        ax_sample.plot(_range, reference, ':', color='black')

    plt.show(block=True)
    model.configuration.update(prev_config)

    return values