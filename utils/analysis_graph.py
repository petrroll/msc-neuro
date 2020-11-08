import numpy as np
import PIL

from IPython.display import display

def reshape_single_as_picture(input, size):
    '''
    Takes a np array and reshapes it to the specified size. Essentially, a transparent wrapper for np.reshape(input, size)
    '''
    return np.reshape(input, size)

def as_single_picture(input, size, outsize=None):
    '''
    Takes a flatten np array (uint8) and its size and returns it as a PIL Image instance, optionally resized to outsize.
    '''
    outsize = outsize if outsize is not None else size 
    input_as_uint8 = input.astype(np.uint8)
    return PIL.Image.fromarray(reshape_single_as_picture(input_as_uint8, size), 'L').resize(outsize, PIL.Image.NEAREST)

import seaborn as sns
import matplotlib.pyplot as plt
def display_as_single_heatmap(input, size):
    '''
    Displays input np array as a heatmap of specified size.
    '''
    display(sns.heatmap(np.reshape(input, size), center=0.0))
    plt.show()

def plot_square_w_as_heatmaps(data, data_extractor, plots=(3, 3)):
    '''
    Takes input `data`, runs `data_extractor(data, i)` for max of (each plot (nrows, ncols), data.shape[1]) and 
    then displays a square heatmap for each of the returned `dta`, arranging them in a (nrows, ncols) grid. Useful
    for displaying set of convolution layer filters.

    E.g.: 
    weights = hsm.networks[0].layers[0].weights
    get_data = lambda w, i: get_gaussian(w[:,i], (SIZE_DOWNSAMPLE, SIZE_DOWNSAMPLE))
    plot_square_w_as_heatmaps(weights, get_data)
    '''
    nrows, ncols = plots
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(16, 16)
    for i in range(min(nrows*ncols, data.shape[1])):
        dta = data_extractor(data, i).flatten()
        size = int(np.sqrt(dta.shape[0]))
        dta = np.reshape(dta, (size, size))
        sns.heatmap(dta, center=0.0, ax=ax[i//ncols][i%ncols])

    plt.show()
        
import math
import numpy as np
import matplotlib.pyplot as plt
def get_metrics_runs(dta, normalize_steps=False, median_mode = False):
    '''
    Get performance metrics (vals_top, vals_bot, vals_mid, vals_bot_sec, steps) for set of runs. Expects pd df{Steps, Run, Value}.

    - normalize_steps: Normalizes step values to 0-1 range.
    - median_mode: Use (90th, 10th, 50th, 25th) percentiles instead of (95th, 05th, 75th, 25th).
    '''  
    by_step = dta.groupby("Step", as_index=False)
    steps = by_step.first()["Step"].values if not normalize_steps else by_step.first()["Step"].values / by_step.first()["Step"].max()
    
    if median_mode:
        vals_top = by_step.quantile(0.90)["Value"].values
        vals_bot = by_step.quantile(0.10)["Value"].values
        vals_mid = by_step.quantile(0.50)["Value"].values
        vals_bot_sec = by_step.quantile(0.25)["Value"].values
    else:
        vals_top = by_step.quantile(0.95)["Value"].values
        vals_bot = by_step.quantile(0.05)["Value"].values
        vals_mid = by_step.quantile(0.75)["Value"].values
        vals_bot_sec = by_step.quantile(0.25)["Value"].values

    return (vals_top, vals_bot, vals_mid, vals_bot_sec, steps)

def summarize_runs(dta, normalize_steps=False, median_mode = False, **kwargs):
    '''
    Gets fully trained performance metrics (mid_percentile_value, top_percentile_value, bottom_percentile_value) for set of runs, expects pd df{Steps, Run, Value}.

    - normalize_steps: Normalizes step values to 0-1 range (can be ignored for this method, here due to backwards compatibility reasons).
    - median_mode: Use (90th, 10th, 50th, 25th) percentiles instead of (95th, 05th, 75th, 25th).
    '''
    (vals_top, vals_bot, vals_mid, _, _) = get_metrics_runs(dta, normalize_steps, median_mode)
    return (vals_mid[-1], vals_top[-1], vals_bot[-1])


def analyse_runs(dta, fig=None, ax=None, second_level_errbar=False, normalize_steps=False, median_mode = False, figsize=None):
    '''
    Visualizes data for set of runs, expects pd df{Steps, Run, Value}.

    Draws a single line (mid percentile) with error bars (bottom, top percentiles) on a figure corresponding to a set of runs for one experiment. 

    - second_level_errbar: Draws second set of smaller error boxes for second bottom percentile.
    - normalize_steps: Normalizes steps to 0-1 range.
    - median_mode: Use (90th, 10th, 50th, 25th) percentiles instead of (95th, 05th, 75th, 25th).
    '''
    assert (fig is None) == (ax is None)
    (vals_top, vals_bot, vals_mid, vals_bot_sec, steps) = get_metrics_runs(dta, normalize_steps, median_mode)

    if figsize is None:
        figsize = (20, 10)

    # Allows externally passed fix, ax so that it can be added to existing ax
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    # Ideally computes for the whole set of experiments & is uniform across all of them 
    # ..even if they have different leghts -> too much work. Eyeball cca 50-> errbars fit.
    err_every = math.ceil(len(vals_mid)/50)

    ax.set_yticks(np.arange(0, 1., 0.02)) 
    ax.yaxis.grid(True)     

    f_line = ax.errorbar(steps, vals_mid, yerr=[(vals_mid-vals_bot), (vals_top-vals_mid)], capsize=4, alpha=0.85, elinewidth=1, errorevery=err_every, lw=1)[0]
    if second_level_errbar: # Draws second set of error boxes at 0.75-0.25
        _ = ax.errorbar(steps, vals_mid, yerr=[(vals_mid-vals_bot_sec), (vals_mid-vals_mid)], capsize=2, alpha=0.85, elinewidth=2, errorevery=err_every, c=f_line.get_color(), lw=1)
    return f_line

def analyse_static_data(dta, limit_steps, fig=None, ax=None, second_level_errbar=False, normalize_steps=False, figsize=None, **kwargs):
    '''
    Visualizes piece of static data, expects (val_top, val_bot, val_mid, val_bot_sec, number_of_epochs) as dta.

    Draws a single line (mid percentile) with error bars (bottom, top percentiles) on a figure corresponding to a set of runs for one experiment. 

    - second_level_errbar: Draws second set of smaller error boxes for second bottom percentile.
    - normalize_steps: Normalizes steps to 0-1 range.
    '''
    assert (fig is None) == (ax is None)
    (val_top, val_bot, val_mid, val_bot_sec, epochs) = dta
    if limit_steps is None:
        limit_steps = (float("-inf"), float("+inf"))

    steps = [x for x in range(-1, epochs-1, 100) if x >= 0 and x >= limit_steps[0] and x <= limit_steps[1]]

    vals_mid = np.array([val_mid for x in steps])
    vals_top = np.array([val_top for x in steps])
    vals_bot = np.array([val_bot for x in steps])
    vals_bot_sec = np.array([val_bot_sec for x in steps])

    if normalize_steps:
        steps = steps / np.max(steps)

    if figsize is None:
        figsize = (20, 10)

    # Allows externally passed fix, ax so that it can be added to existing ax
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    # Ideally computes for the whole set of experiments & is uniform across all of them 
    # ..even if they have different leghts -> too much work. Eyeball cca 50-> errbars fit.
    err_every = math.ceil(len(vals_mid)/50)

    ax.set_yticks(np.arange(0, 1., 0.02)) 
    ax.yaxis.grid(True)     

    f_line = ax.errorbar(steps, vals_mid, yerr=[(vals_mid-vals_bot), (vals_top-vals_mid)], capsize=4, alpha=0.85, elinewidth=1, errorevery=err_every, lw=1)[0]
    if second_level_errbar: # Draws second set of error boxes at 0.75-0.25
        _ = ax.errorbar(steps, vals_mid, yerr=[(vals_mid-vals_bot_sec), (vals_mid-vals_mid)], capsize=2, alpha=0.85, elinewidth=2, errorevery=err_every, c=f_line.get_color(), lw=1)
    return f_line

import utils.data as udata  
def analyse_experiments(experiments, tag, limit_steps=None, experiments_log_in_legend=True, override_legend=None, title=None, enable_legend = True, static_data = [], figsize=(20, 10), **kwargs):
    '''
    Visualizes data for a set of experiments, expects [(experiment_folder, experiment_TB_like_regex), ...]. Returns figure.

    Draws a single figure with a line for each experiment. For line description see `analyse_runs(...)`.

    - limit_steps (float, float)|None: Only display data for steps within bounds (expects absolute step numbers / float(inf))
    - experiments_log_in_legend bool: Show corresponding entries from ./training_data/experiments.txt log file for each experiment in legend.
    - static_data: adds a line based on static values, expects (name_on_legend, data) format. For data format, refer to `analyse_static_data` 
    - ...paramaters of `analyse_runs(...)`
    '''
    fig, ax = plt.subplots(figsize=figsize)
    line_handles = []
    legend_names = []

    for (name, dta) in static_data:
        line_handle = analyse_static_data(dta, limit_steps, fig=fig, ax=ax, figsize=figsize, **kwargs)
        line_handles.append(line_handle)
        legend_names.append(name)

    for i, (folder, regex) in enumerate(experiments):
        dta, logs_num = udata.get_log_data_for_experiment(folder, regex, tag, limit_steps)        
        line_handle = analyse_runs(dta, fig=fig, ax=ax, figsize=figsize, **kwargs)
        
        line_handles.append(line_handle)
        if override_legend is None:
            legend_exp = udata.get_experiment_entries(folder, regex) if experiments_log_in_legend else []
            legend_names.append(f"{regex} ({logs_num} runs) {', '.join(legend_exp)}")
        else:
            legend_names.append(override_legend[i])
    
    if enable_legend:
        ax.legend(line_handles, legend_names)

    ax.set_xlabel("training epoch")
    ax.set_ylabel("mean validation set correlation")
    ax.tick_params(labelleft=True, labelright=True,)
    if title: ax.set_title(title)

    fig = plt.gcf()
    plt.show()
    return fig

def summarize_experiments(experiments, tag, limit_steps=None, experiments_log_in_legend=True, override_legend=None, title=None, **kwargs):
    '''
    Prints trained performance summaries for set of experiments, expects [(experiment_folder, experiment_TB_like_regex), ...].

    Prints a single summary for each experiment. For summary description see `summarize_runs(...)`.

    - limit_steps (float, float)|None: Only shows data for steps within bounds (expects absolute step numbers / float(inf))
    - experiments_log_in_legend bool: Show corresponding entries from experiments.txt log file.
    - ...paramaters of `summarize_runs(...)`
    '''
    for i, (folder, regex) in enumerate(experiments):
        dta, logs_num = udata.get_log_data_for_experiment(folder, regex, tag, limit_steps)        
        runs_summary = summarize_runs(dta, **kwargs)
        
        if override_legend is None:
            legend_exp = udata.get_experiment_entries(folder, regex) if experiments_log_in_legend else []
            legend = f"{regex} ({logs_num} runs) {', '.join(legend_exp)}"
        else:
            legend = override_legend[i]

        print(f"{legend}: {tuple(map(lambda x: round(x, 2), runs_summary))}")

