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
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as CNP
    from .generic_model import GenericModel

from io import TextIOWrapper
import matplotlib.pyplot as plt


from math import ceil
import copy



def offset_array(array, offset):
    """Offsets an array by the given amount.

    Args:
        array (list): array to be changed.
        offset (int): offset to apply to the given array.
    """
    if (offset > 0):
        array[offset:] = array[:-offset]
        array[:offset] = array[0]
    elif (offset < 0):
        array[:offset] = array[-offset:]
        array[offset:] = array[offset-1]

def get_best_parameters(params, log_diff, save_percentage):
    "Retuns the best `save_percentage`% `params` of the simulations given their `log_diff` with real data." 
    save_count: int = ceil(log_diff.size*save_percentage*0.01)
    saved_params = CNP.zeros((save_count, 1), dtype=params.dtype)
    saved_log_diff = CNP.zeros((save_count, 1), dtype=CNP.float64)

    if save_count == log_diff.size:
        log_diff_index_sorted = CNP.argsort(log_diff, 0)
    else:
        log_diff_index_sorted = CNP.argpartition(log_diff, save_count, 0)[0:save_count]
    
    saved_params[:] = CNP.take(params, log_diff_index_sorted)
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
        # import numpy as np
        # if np != CNP:
        #     _params = CNP.asnumpy(params)
        # else:
        #     _params = params
        # np.savetxt(file_out, params.T, delimiter=',', comments='', header=",".join(params_names) if execution_number==0 else "")
        import regex as re

        _merged = params
        file_out.write(",".join(_merged.dtype.names)+"\n")
        file_out.write(re.sub(r"\[|\]|\(|\)| ", "", CNP.array_str(_merged)))


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
        # TODO: check for cupy
        # import numpy as np
        import regex as re
        # if np != CNP:
        #     np.savetxt(file_out, CNP.asnumpy(CNP.concatenate((log_diff, params), 1)) , delimiter=',', comments='', header=",".join(["log_diff", *params_names]) if execution_number==0 else "")
        # else:
        _log_diff = log_diff
        _params = params
        _log_diff = CNP.asarray(log_diff, dtype=[("log_diff", CNP.float64)])
        _merged = CNP.zeros((_params.size,1), dtype=[*_log_diff.dtype.descr, *_params.dtype.descr])
        _merged["log_diff"] = _log_diff
        for k in _params.dtype.names:
            _merged[k] = _params[k]

        if execution_number==0:
            file_out.write(",".join(_merged.dtype.names)+"\n")
        else:
            file_out.write("\n")
        file_out.write(re.sub(r"\[|\]|\(|\)| ", "", CNP.array_str(_merged)))
            

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


def get_model_sample_trajectory(model: GenericModel, *args, **kargs):
    """Executes the model with `n_simulations = 1` and `n_executions = 1`.
    Returns all the intermediate states and the parameters.

    Args:
        model (GenericModel): Model to execute.

    Returns:
        (list[list[float]], list[list[float]]): Tuple of all states history and corresponding params.
    """
    reference_mask = CNP.array([model.compartment_name_to_index[c] for c in model.configuration["reference"]["compartments"]])
    configuration = copy.deepcopy(model.configuration)

    from . import GenericModel
    inner_model = GenericModel(configuration)
    inner_model.evolve = model.evolve

    inner_model.configuration["simulation"]["n_simulations"] = 1
    inner_model.configuration["simulation"]["n_executions"] = 1
    
    inner_model.populate_model_parameters(**kargs)
    inner_model.populate_model_compartments(**kargs)
    saved_state = CNP.zeros((inner_model.configuration["simulation"]["n_steps"], inner_model.state.shape[0]))

    def inner(_model_, step, reference, reference_mask, *args, **kargs):
        _model_.evolve(_model_, step, *args, **kargs)
        saved_state[step] = _model_.state[:, 0]

    def outer(_model_, *args, **kargs):
        ...
        
    inner_model._internal_run_(
        inner, (reference_mask, *args), 
        outer, (), 
        None, None,
        *args, exclude_pupulate=True, **kargs
    )
        
    offset_array(saved_state, inner_model.reference_offset[0])
    return saved_state.T, inner_model.params[0]

def get_model_sample_trajectory_with_diff_to_reference(model: GenericModel, reference, *args, **kargs):
    """Executes the model with `n_simulations = 1` and `n_executions = 1`.
    Returns all the intermediate states and the parameters.

    Args:
        model (GenericModel): Model to execute.
        reference (list): Reference to compare with.

    Returns:
        (list[list[float]], list[list[float]], float): Tuple of all states history and corresponding params and the difference.
    """
    reference_mask = CNP.array([model.compartment_name_to_index[c] for c in model.configuration["reference"]["compartments"]])
    configuration = copy.deepcopy(model.configuration)
    
    from . import GenericModel
    inner_model = GenericModel(configuration)
    inner_model.evolve = model.evolve
    reference_mask = CNP.array([inner_model.compartment_name_to_index[c] for c in inner_model.configuration["reference"]["compartments"]])

    inner_model.configuration["simulation"]["n_simulations"] = 1
    inner_model.configuration["simulation"]["n_executions"] = 1
    inner_model.N_STEPS = inner_model.configuration["simulation"]["n_steps"]
    
    inner_model.populate_model_parameters(**kargs)
    inner_model.populate_model_compartments(**kargs)
    saved_state = CNP.zeros((inner_model.configuration["simulation"]["n_steps"], inner_model.state.shape[0]))
    log_diff = CNP.zeros((1))

    def inner(_model_, step, reference, reference_mask, *args, **kargs):
        _model_.evolve(_model_, step, *args, **kargs)
        log_diff[0] += inner_model.get_diff(step, reference, reference_mask)[0]
        saved_state[step] = _model_.state[:, 0]

    def outer(_model_, *args, **kargs):
        ...
        
    inner_model._internal_run_(
        inner, (reference_mask, *args), 
        outer, (), 
        reference, None,
        *args, exclude_pupulate=True, **kargs
    )
        
    offset_array(saved_state, inner_model.reference_offset[0])
    return saved_state.T, inner_model.params[0], log_diff


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
        values = CNP.copy(values[sorter])
        sample_weight = sample_weight[sorter]

    weighted_quantiles = CNP.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= CNP.sum(sample_weight)
    return CNP.interp(quantiles, weighted_quantiles, values)


def get_percentiles_from_results(model: GenericModel, results, p_minor=5, p_max=95, weights=None, *args, **kargs):
    """Returns an array of percentils `p_minor=5`, median and `p_max=95` of the given model and results.

    Args:
        model (GenericModel): Model used to generate the `results`.
        results (list[list[float]]): Result parameters of `model` execution.
        p_minor (int, optional): Smaller percentile. Defaults to 5.
        p_max (int, optional): Bigger percentile. Defaults to 95.
        weights (list[float]|None): Results weights. Defaults to None.

    Returns:
        (list[int, int, list[float]]): First index represents the reference defined in `reference.compartments`. \
            Second index represents  `p_minor`, median or `p_max=`. Final represents the step in the simulation.
    """
    reference_mask = CNP.array([model.compartment_name_to_index[c] for c in model.configuration["reference"]["compartments"]])
    
    results_no_diff = results[1:]
    results_percentiles = CNP.zeros((reference_mask.shape[0], 3, model.configuration["simulation"]["n_steps"]))
    
    prev_config = copy.deepcopy(model.configuration)
    configuration = copy.deepcopy(model.configuration)
    configuration["simulation"]["n_simulations"] = results.shape[1]
    configuration["simulation"]["n_executions"] = 1
    
    from . import GenericModel
    inner_model = GenericModel(configuration)
    inner_model.evolve = model.evolve

    inner_model.populate_model_parameters(**kargs)
    for k in inner_model.params.dtype.names:
        inner_model.params[k][:] = results_no_diff[inner_model.param_to_index[k]]

    inner_model.populate_model_compartments(**kargs)

    storage = CNP.zeros((configuration["simulation"]["n_steps"], configuration["simulation"]["n_simulations"]))
    _range = CNP.arange(configuration["simulation"]["n_simulations"])
    
    def inner(_model_, step, reference, reference_mask, *args, **kargs):
        _model_.evolve(_model_, step, *args, **kargs)
        aux = CNP.take(_model_.state, reference_mask, 0)
        

        # TODO: improve this so that storage is not bigger than (offset.max - offset.min, n_simulations) 
        # This line is a bit complex, what it does:
        # Save in the storage at the correct time for each simulation the values of the aux (=values of state to compare with reference) 
        # in the same order so that each simulation is treated independently until all simulations stop
        storage[CNP.clip(_model_.reference_offset + step, 0, _model_.configuration["simulation"]["n_steps"]-1),
             _range] += aux[0][_range] * (_model_.reference_offset + step < _model_.configuration["simulation"]["n_steps"])

        
    def outer(_model_, *args, **kargs):
        ...
        
    inner_model._internal_run_(
        inner, (reference_mask, *args), 
        outer, (), 
        None, None,
        *args, exclude_pupulate=True, **kargs
    )

    if weights is not None:
        percentile = lambda x,p: weighted_quantile(x, p, weights, False, False)
    else:
        percentile = lambda x,p: CNP.percentile(x, p)

    for step in range(model.configuration["simulation"]["n_steps"]):
        sort = CNP.sort(storage[step])
        results_percentiles[:, 0, step] += percentile(sort, p_minor)
        results_percentiles[:, 1, step] += percentile(sort, 50)
        results_percentiles[:, 2, step] += percentile(sort, p_max)

    return results_percentiles


def auto_adjust_model_params(model: GenericModel, results, weights=None, params=None):
    """Adjusts limits of model params. If `params` is specified only those are adjusted.

    Args:
        model (GenericModel): Model to optimize.
        results (list[list[float]]): Results from running the model.
        weights (list[float], optional): Results weights. Defaults to None.
        params (list[str], optional): Names of params to optimice. Defaults to None.
    """
    if weights is not None:
        percentile = lambda x,p: weighted_quantile(x, p, weights, False, False)
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

        TYPE = M.get("type", "float64")
        if "int" in TYPE:
            MIN = _5 - 1 
            MAX = _95 + 1
        else :
            MIN = _5 * 0.5
            MAX = _95 * 2
        


        min_probable_value = _5 - distM if distM != 0 else MAX
        max_probable_value = _95 + distm if distm != 0 else MIN

        # print(f"""{c}:
        #     min: {min_probable_value},\t {0.9 * (M["min"]*4+_5)/5},\t {M.get("min_limit", None)}
        #     max: {max_probable_value},\t {1.1 * (M["max"]*4+_95)/5},\t {M.get("max_limit", None)}
        #     """)
        
        # TODO: make a gaussian kernel to infer when probability is 0 and assing as min or max
        MAX = M.get("max_limit", None)
        MIN = M.get("min_limit", None)
        if (MAX is not None) or (MIN is not None):
            model.configuration["params"][c].update({
                "min" : CNP.clip(min(min_probable_value, 0.9 * (M["min"]*2+_5)/3), MIN, MAX),
                "max" : CNP.clip(max(max_probable_value, 1.1 * (M["max"]*2+_95)/3), MIN, MAX)
            })
        else:
            model.configuration["params"][c].update({
                "min" :min(min_probable_value, 0.9 * (M["min"]*2+_5)/3),
                "max" :max(max_probable_value, 1.1 * (M["max"]*2+_95)/3)
            })
        

def get_trajecty_selector(model: GenericModel, results, weights, reference=None, *args, show_only_reference=False):
    """Creates an interactive plot and histograms of results. When a histogram is clicked the value of
    that parameter changes to the selected value.

    Args:
        model (GenericModel): Model used for the trajectory.
        results (list[list[float]]): Results from running the model.
        weights (list[float], optional): Results weights. Defaults to None.
        reference (list[list[float]], optional): If give, is printed to the trajectory. Defaults to None.
        show_only_reference (boolean, optional): If `True` only the values used to compare with the reference are ploted. Defaults to False.

    Returns:
        (dict[str, float]): Dictionary with the manually selected params.
    """
    prev_config = copy.deepcopy(model.configuration)
    fig_sample, ax_sample = plt.subplots()
    
    _range = CNP.arange(model.configuration["simulation"]["n_steps"])
    
    # Params used for the trajectory are saved here. This is returned
    values = {}
    compartments_ploted = []
    
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
        values.update({p:(_5+_50)/2})

        # Define a picker por the param ax
        def picker_builder(param, vline):
            def picker(self, event):
                x = event.xdata
                # Update values and vline xdata
                values.update({param:x})
                vline.set_xdata([x,x])
                # Update trajectory
                data, diff = update()
                ax.set_title(f"diff={diff}")
                for compartment, line in zip(compartments_ploted, sample_lines):
                    if show_only_reference and reference is None:
                        continue
                    else:
                        line.set_ydata(data[model.compartment_name_to_index[compartment]])
                
                fig_sample.canvas.draw_idle()
                fig.canvas.draw_idle()
                return True, {}
            return picker
            
        line.set_picker(picker_builder(p, line))
        ax.set_xlim(xlim)

    def update():
        sample, _, diff = get_model_sample_trajectory_with_diff_to_reference(model, reference, *args, **values)
        return sample, diff

    sample, diff = update()
    list_of_sample_lines = []

    try:
        if show_only_reference and reference is not None:
            for k,i in model.compartment_name_to_index.items():
                if k in model.configuration["reference"]["compartments"]:
                    compartments_ploted.append(k)
                    list_of_sample_lines.append(_range)
                    list_of_sample_lines.append(sample[i])
                    list_of_sample_lines.append('-')

        else: 
            raise KeyError
    except KeyError:
        compartments_ploted = list(model.compartment_name_to_index.keys())
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