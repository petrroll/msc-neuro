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