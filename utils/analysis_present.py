import numpy as np
import utils.analysis as uan

def print_stat_matrix(name, stat):
    '''
    Print a matrix stat: "{name}: {mean(stat)} \n {stat}"
    '''
    print(name + ": " + str(np.mean(stat)))
    print(stat)
    
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

def print_stat_scalar(name, stat):
    '''
    Print a scalar stat "{name}: {stat}"
    '''
    print(name + ": " + str(stat))

def evaluate_output_neuron(predicted, golden, corr):  
    '''
    Evaluates a single output neuron.

    Expects predicted (np.array), golden (np.array), and correlation (float) data for a single neuron. 
    Prints correlation, standard deviation on predicted data, golden data, means on both, and first 15 examples of each.
    '''
    print_stat_scalar("Corr", corr)

    print_stat_scalar("STD result", np.std(predicted, axis=0))
    print_stat_scalar("STD golden", np.std(golden, axis=0))
    
    print_stat_scalar("Mean result", np.mean(predicted, axis=0))
    print_stat_scalar("Mean golden", np.mean(golden, axis=0))

    print("Example result", predicted[:15])
    print("Example golden", golden[:15])
    
def evaluate_best_corr_neuron(predicted, golden, data_filters=None):
    '''
    Expects predicted, golden, and data filters for the whole model. Identifies highest correlation output neuron, prints its evaluation.

    Uses `evaluate_output_neuron(...)` for the best neuron evaluation.
    '''
    data_filters = data_filters if data_filters is not None else np.ones(golden.shape)
    
    corr = uan.get_correlation(predicted, golden, data_filters)
    corr[np.isnan(corr)] = 0
    
    i = np.nanargmax(corr)
    print_stat_scalar("Argmax i", i)
    
    mask = data_filters[:,i] == 1 
    evaluate_output_neuron(predicted[mask, i], golden[mask, i], corr[i])

def print_summary_stats(data):
    '''
    Prints summary (min, max, mean, median, std) for a numpy array.
    '''
    print(f"Min/Max:{np.min(data)}/{np.max(data)} Mean/Median:{np.mean(data)}/{np.median(data)} Std:{np.std(data)}")