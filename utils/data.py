import numpy as np 

def load_data(type, region):
    # `raw_validation_set.npy` is multiple tries non-averaged `validation_set.npy`
    return (
        (np.squeeze(np.load(f'./Data/region{region}/{type}_inputs.npy'))).astype(np.float64), # data is in float64 format: 0.0-0.000255
        np.squeeze(np.load(f'./Data/region{region}/{type}_set.npy')).astype(np.float64)
    )

def merge_data_and_get_mask(inputs, outputs):
    # Assumes output is flat vector
    # Assumes result neurons for individual outputs are independent i.e. first result neuron of first output
    # ..is different from first neuron of second output
    assert len(inputs) == len(outputs)
    
    total_len = 0
    total_output_dim = 0
    for i in range(len(inputs)):
        assert inputs[i].shape[0] == outputs[i].shape[0]
        total_len += inputs[i].shape[0]
        total_output_dim += outputs[i].shape[1]
        
    merged_inputs = np.concatenate(inputs, axis=0)
    merged_outputs = np.zeros([total_len, total_output_dim], np.float64)
    merged_outputs_mask = np.zeros([total_len, total_output_dim], np.float64)

    startI = 0
    startD = 0
    start_ends = []
    for output in outputs:
        length, dim = output.shape
        endD, endI = startD+dim, startI+length
        
        merged_outputs[startI:endI, startD:endD] = output
        merged_outputs_mask[startI:endI, startD:endD] = 1.0
        
        start_ends.append(((startI, endI), (startD, endD)))
        
        startI = endI
        startD = endD
    return merged_inputs, merged_outputs, merged_outputs_mask

def load_data_multiple(indexes, dta_type, input_processor=None, output_processor=None):
    dta_input = []
    dta_output = []
    for i in indexes:
        input_tr, output_tr = load_data(dta_type, i)

        input_tr = input_tr if input_processor is None else input_processor(input_tr)
        output_tr = output_tr if output_processor is None else output_processor(output_tr)

        dta_input.append(input_tr)
        dta_output.append(output_tr)
    
    return merge_data_and_get_mask(dta_input, dta_output)

def normalize_mean_std(dta):
    return (dta - np.mean(dta)) / np.std(dta)

def normalize_std(dta):
    return dta / np.std(dta)

import tensorflow as tf
import os

def get_tag_values(file_path, tag_name):
    '''
    Return value of a specified tag for each step from a TF Event file.
    '''
    steps = []
    tag_values = []
    for e in tf.train.summary_iterator(file_path):
        if e.WhichOneof("what") == "summary":
            steps.append(e.step)
            for v in e.summary.value:
                if v.tag == tag_name:
                    tag_values.append(v.simple_value)
                    
    assert len(steps) == len(tag_values)
    return (steps, tag_values)

import re
def get_file_paths_filtered(root, regex=r".*"):
    '''
    Returns paths to all files conforming to specified regex.
    '''
    pattern = re.compile(regex)
    file_paths = []
    
    for root, _, files in os.walk(root):
        for file in files:
            file_path = f"{root}/{file}" 
            if pattern.search(file_path):
                file_paths.append(file_path)
                
    return file_paths

def get_experiment_entries(regex=r".*", only_suffix_after_matched=True, path="./training_data/experiments.txt"):
    '''
    Returns all entries in experiments.txt log for regex.

    - only_after_matched: Only returns suffixes after the matched part (removing .* near the end is important for this to work)
    - Automatically removes ".*test" from the end of the regex.
    - Doesn't take root & assumes the regex identifies desired experiment uniquely across experiment folders, etc.
    '''
    regex = regex[:-6] if regex[-6:] == ".*test" else regex     # Ignore matching .*test suffix
    regex = f"^[^#].*{regex}"                                   # Ignore comments, theoretically this will make it match everything in case of e.g. empty regex but it shouldn't be a real problem
    pattern = re.compile(regex)
    
    with open("./training_data/experiments.txt", "r") as f:
        if only_suffix_after_matched:
            # `pattern.search(line).span()[1]` gets you the index of the end of the matched substring
            return [ line[pattern.search(line).span()[1]:].strip() for line in f if pattern.search(line) ]   
        else:
            return [ line.strip() for line in f if pattern.search(line) ]   

import pandas as pd
def load_data_from_event_files(file_paths, tag):
    '''
    Loads data from specified TF Event files to a pd df{Steps, Run, Value}, names runs {0, 1, ...}.
    '''
    steps, values, runs = [], [], []
    for i in range(len(file_paths)):
        log_steps, log_values = get_tag_values(file_paths[i], tag)
    
        steps += log_steps
        values += log_values
        runs += [i]*len(log_values)
        
    return pd.DataFrame({
        'Step':steps, 
        'Run':runs, 
        'Value':values
    })

def get_log_data_for_experiment(folder, regex, tag, limit_steps=None):
    '''
    Returns the log data and the number of used logs for an experiment specied by a logs folder and a regex 
    filtered to include only steps conforming specified bounds.
    '''
    file_paths = get_file_paths_filtered(folder, regex)
    dta = load_data_from_event_files(file_paths, tag)
    if limit_steps is not None:     # Steps are between limit_steps (min, max) values
        dta = dta.loc[(dta['Step'] >= limit_steps[0]) & (dta['Step'] <= limit_steps[1])]

    return (dta, len(file_paths))
