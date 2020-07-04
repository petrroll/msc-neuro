import numpy as np
import utils.analysis as uan

from IPython.display import display, Image

def print_stat_matrix(string, stat):
    display(string + ": " + str(np.mean(stat)))
    display(stat)
    
def evaluate_all(hsm, input, golden, data_filters=None):
    dta_len, _ = input.shape
    data_filters = data_filters if data_filters is not None else np.ones(golden.shape)

    nuladj_eval = hsm.eval_models(input_data=input, output_data=golden, data_indxs=np.array(range(dta_len)),
                           data_filters=data_filters, 
                           nulladjusted=False) # nulladjusted=True doesn't work well with data_filters
    print_stat_matrix("Eval ", nuladj_eval)
    
    result = hsm.generate_prediction(input)
    result = np.multiply(result, data_filters)
    
    corr = uan.get_correlation(result, golden, data_filters)
    corr[np.isnan(corr)] = 0
    print_stat_matrix("Corr", corr)
    
    std_result, std_golden = np.std(result, axis=0), np.std(golden, axis=0)
    print_stat_matrix("STD result", std_result)
    print_stat_matrix("STD golden", std_golden)  
    
    return result, nuladj_eval, corr

def print_stat_scalar(string, stat):
    display(string + ": " + str(stat))

def evaluate_output_neuron(result, golden, corr):  
    print_stat_scalar("Corr", corr)

    print_stat_scalar("STD result", np.std(result, axis=0))
    print_stat_scalar("STD golden", np.std(golden, axis=0))
    
    print_stat_scalar("Mean result", np.mean(result, axis=0))
    print_stat_scalar("Mean golden", np.mean(golden, axis=0))

    display("Example result", result[:15])
    display("Example golden", golden[:15])
    
def evaluate_best_corr_neuron(result, golden, data_filters=None):
    data_filters = data_filters if data_filters is not None else np.ones(golden.shape)
    
    corr = uan.get_correlation(result, golden, data_filters)
    corr[np.isnan(corr)] = 0
    
    i = np.nanargmax(corr)
    print_stat_scalar("Argmax i", i)
    
    mask = data_filters[:,i] == 1 
    evaluate_output_neuron(result[mask, i], golden[mask, i], corr[i])

import PIL
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
