import numpy as np
import PIL

from IPython.display import display

def reshape_single_as_picture(input, size):
    return np.reshape(input, size)

def as_single_picture(input, size, outsize=None):
    outsize = outsize if outsize is not None else size 
    input_as_uint8 = input.astype(np.uint8)
    return PIL.Image.fromarray(reshape_single_as_picture(input_as_uint8, size), 'L').resize(outsize, PIL.Image.NEAREST)

import seaborn as sns
import matplotlib.pyplot as plt
def display_as_single_heatmap(input, size):
    display(sns.heatmap(np.reshape(input, size), center=0.0))
    plt.show()

def plot_square_w_as_heatmaps(w, get_data, plots=(3, 3)):
    nrows, ncols = plots
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(16, 16)
    for i in range(min(nrows*ncols, w.shape[1])):
        dta = get_data(w, i).flatten()
        size = int(np.sqrt(dta.shape[0]))
        dta = np.reshape(dta, (size, size))
        sns.heatmap(dta, center=0.0, ax=ax[i//ncols][i%ncols])

    plt.show()
        
import math
import numpy as np
import matplotlib.pyplot as plt
def analyse_runs(dta, fig=None, ax=None, second_level_errbar=False):
    '''
    Visualizes data for set of runs, expects pd df{Steps, Run, Value}.
    '''
    assert (fig is None) == (ax is None)
    
    by_step = dta.groupby("Step", as_index=False)

    steps = by_step.first()["Step"].values
    vals_95 = by_step.quantile(0.95)["Value"].values
    vals_05 = by_step.quantile(0.05)["Value"].values
    vals_75 = by_step.quantile(0.75)["Value"].values
    vals_25 = by_step.quantile(0.25)["Value"].values

    # Allows externally passed fix, ax so that it can be added to existing ax
    if fig is None:
        fig, ax = plt.subplots(figsize=(20,10))
        
    # Ideally computes for the whole set of experiments & is uniform across all of them 
    # ..even if they have different leghts -> too much work. Eyeball cca 50-> errbars fit.
    err_every = math.ceil(len(vals_05)//50)

    ax.set_yticks(np.arange(0, 1., 0.025))   
    ax.yaxis.grid(True)   
    f_line = ax.errorbar(steps, vals_75, yerr=[(vals_75-vals_05), (vals_95-vals_75)], capsize=8, alpha=0.75, elinewidth=2, errorevery=err_every)[0]
    if second_level_errbar: # Draws second set of error boxes at 0.75-0.25
        _ = ax.errorbar(steps, vals_75, yerr=[(vals_75-vals_25), (vals_75-vals_75)], capsize=4, alpha=0.75, elinewidth=3, errorevery=err_every, c=f_line.get_color())
    return f_line

import utils.data as udata  
def analyse_experiments(experiments, tag, **kwargs):
    '''
    Visualizes data for set of experiments, expects [(experiment_folder, experiment_TB_like_regex), ...].
    '''
    fig, ax = plt.subplots(figsize=(20,10))
    handles = []
    for (folder, regex) in experiments:
        file_paths = udata.get_file_paths_for_experiment(folder, regex)
        dta = udata.load_data_from_event_files(file_paths, tag)
        p = analyse_runs(dta, fig=fig, ax=ax, **kwargs)
        handles.append(p)
        
    legend_list = list(map(lambda x: x[1], experiments))
    ax.legend(handles, legend_list)
    plt.show()
