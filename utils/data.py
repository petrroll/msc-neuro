import numpy as np 

def load_data(type, region):
    '''
    Loads data for a specified region (numeric) and type (training|validation) as float64 arrays. Returns a tuple of (input, output).
    '''
    # `raw_validation_set.npy` is multiple tries non-averaged `validation_set.npy`
    return (
        (np.squeeze(np.load(f'./Data/region{region}/{type}_inputs.npy'))).astype(np.float64), # data is in float64 format: 0.0-0.000255
        np.squeeze(np.load(f'./Data/region{region}/{type}_set.npy')).astype(np.float64)
    )

def merge_data_and_get_mask(inputs, outputs):
    '''
    Transforms individual data sets into a shared and returns (input, output, mask).

    Mask conforms to filter_data of NDN3 library. Assumes input have the same dimensionality.

    Expects inputs and outputs in a form of [[set_1_data], [set_2_data], ...], produces [set_1_data, set_2_data, ...]. Concatenates inputs,
    widens outputs so that each set's outputs can be next to each other, and concatenates outputs in a way that set_1 outputs are on output
    dimensions 0 to |set_1_output_dims| for datapoints 0 to len(set_1) and zeroes elsewhere, outputs for set_2 are on dimensions |set_1_output_dims| to 
    |set_1_output_dims|+|set_2_output_dims| for datapoints len(set_1) to len(set_1)+len(set_2) etc. Also prepares mask that has ones on active 
    output positions.

    Example:
        merge_data_and_get_mask(
            [[1, 2], [2, 3, 4]], 
            [[[10, 11], [20, 22]], [200, 300, 400]]) ->
        (
            [1, 2, 2, 3, 4], 
            [[10, 11, 0], [20, 22, 0], [0, 0, 200], [0, 0, 300], [0, 0, 400]], 
            [[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]], 
        )
    '''
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

def load_data_multiple(regions, type, input_processor=None, output_processor=None):
    '''
    Loads multiple regions, processes them, and merges into one.

    - regions, type: correspond to `load_data(...)`
    - input/output_processor: processor functions [float] -> [float] that run for input and output data
    
    Note:
        Merging done through `merge_data_and_get_mask(...)`, see its documentation for more information.
    '''
    dta_input = []
    dta_output = []
    for i in regions:
        input_tr, output_tr = load_data(type, i)

        input_tr = input_tr if input_processor is None else input_processor(input_tr)
        output_tr = output_tr if output_processor is None else output_processor(output_tr)

        dta_input.append(input_tr)
        dta_output.append(output_tr)
    
    return merge_data_and_get_mask(dta_input, dta_output)

def normalize_mean_std(dta):
    '''
    Normalizes the input np array to 0 mean and 1 standard deviation.
    '''
    return (dta - np.mean(dta)) / np.std(dta)

def normalize_std(dta):
    '''
    Normalizes the input np array to 1 standard deviation.
    '''
    return dta / np.std(dta)

import tensorflow as tf
import os

def get_tag_values(file_path, tag_name):
    '''
    Gets the values of a specified tag for each step from a TF Event file. Returns (steps, tag_values).
    '''
    steps = []
    tag_values = []
    for e in tf.train.summary_iterator(file_path):
        if e.WhichOneof("what") == "summary":
            for v in e.summary.value:
                if v.tag == tag_name:
                    steps.append(e.step)
                    tag_values.append(v.simple_value)
                    
    assert len(steps) == len(tag_values)
    return (steps, tag_values)

import re
def get_file_paths_filtered(root, regex=r".*"):
    '''
    Returns paths to all files in a root folder with "{root}/{file}" conforming to the specified regex.
    '''
    pattern = re.compile(regex)
    file_paths = []
    
    for root, _, files in os.walk(root):
        for file in files:
            file_path = f"{root}/{file}" 
            if pattern.search(file_path):
                file_paths.append(file_path)
                
    return file_paths

def get_experiment_entries(folder="", regex=r".*", only_suffix_after_matched=True, path="./training_data/experiments.txt"):
    '''
    Returns all entries in ./training_data/experiments.txt log for an experiment specified by a folder and filter regex.

    - only_after_matched: Only returns suffixes after the matched part (removing .* near the end of the regex is important for this to work)

    Notes:
    - folder might begin with "(./)?training_data/logs/" that needs to get removed to match experiments.txt entries
    - Ignores comments starting with `#`
    - Automatically removes ".*test" from the end of the regex.
    '''
    def is_not_comment(line):
        return len(line) > 0 and line[0] != "#"

    regex = regex[:-6] if regex[-6:] == ".*test" else regex # Remove matching .*test suffix from regex (not present on entries)
    folder = folder.split("training_data/logs/")[-1]        # Remove matching logs folder prefix, take substring after it (not present on entries) (FIXME: This should be better abstracted)
    folder = folder if folder[-1] == "/" else f"{folder}/"  # Need to potentially append a "/" to make sure the experiment folder part is used to actually match experiment folder as part of the path an nothing else (FIXME: Not robust)
    regex = f"{re.escape(folder)}.*{regex}"                 # Include folder: theoretically will make it match everything in case of e.g. empty regex but it shouldn't be a real problem
    pattern = re.compile(regex)

    with open("./training_data/experiments.txt", "r") as f:
        if only_suffix_after_matched:
            # `pattern.search(line).span()[1]` gets you the index of the end of the matched substring
            return [ line[pattern.search(line).span()[1]:].strip() for line in f if pattern.search(line) and is_not_comment(line) ]   
        else:
            return [ line.strip() for line in f if pattern.search(line) and is_not_comment(line) ]   

import pandas as pd
def load_data_from_event_files(file_paths, tag):
    '''
    Loads data from specified TF Event files to a pd df{Steps, Run, Value}. 
    
    Sequentially names runs {0, 1, ...}, one for each path in `file_paths`. 
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
    Returns experiment metrics (`tag`) pd df{Steps, Run, Value} for an experiment specified as TF Event logs `folder` and a filter `regex`. 
    Treats each TF Event log as separate run.

    Optionally limits returned data (pd, number_of_runs) to a range of steps (min, max) for each run.
    '''
    file_paths = get_file_paths_filtered(folder, regex)
    dta = load_data_from_event_files(file_paths, tag)
    if limit_steps is not None:     # Steps are between limit_steps (min, max) values
        dta = dta.loc[(dta['Step'] >= limit_steps[0]) & (dta['Step'] <= limit_steps[1])]

    return (dta, len(file_paths))
