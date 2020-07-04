import numpy as np
import utils.analysis as uan

def print_stat_matrix(string, stat):
    print(string + ": " + str(np.mean(stat)))
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

def print_stat_scalar(string, stat):
    print(string + ": " + str(stat))

def evaluate_output_neuron(result, golden, corr):  
    print_stat_scalar("Corr", corr)

    print_stat_scalar("STD result", np.std(result, axis=0))
    print_stat_scalar("STD golden", np.std(golden, axis=0))
    
    print_stat_scalar("Mean result", np.mean(result, axis=0))
    print_stat_scalar("Mean golden", np.mean(golden, axis=0))

    print("Example result", result[:15])
    print("Example golden", golden[:15])
    
def evaluate_best_corr_neuron(result, golden, data_filters=None):
    data_filters = data_filters if data_filters is not None else np.ones(golden.shape)
    
    corr = uan.get_correlation(result, golden, data_filters)
    corr[np.isnan(corr)] = 0
    
    i = np.nanargmax(corr)
    print_stat_scalar("Argmax i", i)
    
    mask = data_filters[:,i] == 1 
    evaluate_output_neuron(result[mask, i], golden[mask, i], corr[i])

def print_summary_stats(dta):
    print(f"Min/Max:{np.min(dta)}/{np.max(dta)} Mean/Median:{np.mean(dta)}/{np.median(dta)} Std:{np.std(dta)}")