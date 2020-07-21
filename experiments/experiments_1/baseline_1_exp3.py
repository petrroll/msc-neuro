import numpy as np

import os
import sys
import math

from datetime import datetime
from importlib import reload
from pprint import pprint

from platform import python_version
print(python_version())

sys.path.append(os.getcwd())

import NDN3.NDNutils as NDNutils
import NDN3.NDN as NDN

import utils.data as udata
import utils.network as unet
import utils.analysis as uas
import utils.analysis_present as uasp

#
# 1
# Even DoG has softplus non-linearity.
#
name = 'baselineshort_exp31_1mult1e6_Dog9xN0x2N_SpxSpxSp_Nonorm_1e3x16x5000'
exp_folder = "experiments_1"
exp = "bs1_exp31"

def get_hsm_params_custom(input, output, i):
    _, output_shape = output.shape
    _, input_shape = input.shape
    pprint(f"in: {input_shape} out: {output_shape}")
        
    intput_w, input_h = int(math.sqrt(input_shape)), int(math.sqrt(input_shape))
    hsm_params = NDNutils.ffnetwork_params(
        verbose=False,
        input_dims=[1, intput_w, input_h], 
        layer_sizes=[9, int(0.2*output_shape), output_shape], # paper: 9, 0.2*output_shape
        ei_layers=[None, None, None],
        normalization=[0, 0, 0], 
        layer_types=['diff_of_gaussians','normal','normal'],
        act_funcs=['softplus', 'softplus','softplus'],
        reg_list={
            'd2x':[None,None,None],
            'l1':[None,None,None],
            'max':[None,None,None]})
    hsm_params['weights_initializers']=['random','normal','normal']

    return hsm_params

def get_training_params():
    return {'batch_size': 16, 'use_gpu': False, 'epochs_summary': 100, 'epochs_training': 5000, 'learning_rate': 0.1e-3}

input_tr_processed, output_tr, output_tr_mask = udata.load_data_multiple(
    [1], 'training', lambda x: x*1_000_000)
input_val_processed, output_val, output_val_mask = udata.load_data_multiple(
    [1], 'validation', lambda x: x*1_000_000)

for i in range(10):
    seed = i
       
    hsm_params = get_hsm_params_custom(input_tr_processed, output_tr, i)
    pprint(hsm_params)
    hsm, input_tuple = unet.get_network(
        input_tr_processed, output_tr,
        'adam', 
        get_training_params(),
        hsm_params,
        'poisson',
        input_val_processed, output_val,
        output_tr_mask, output_val_mask,
        f"{name}__{i}", seed,
        
    )
    hsm.log_correlation = 'zero-NaNs'
        
    (input, output, train_indxs, test_indxs, data_filters, larg, opt_params, name_str) = input_tuple
    hsm.train(
        input_data=input, 
        output_data=output, 
        train_indxs=train_indxs, 
        test_indxs=test_indxs, 
        data_filters=data_filters,
        learning_alg=larg, 
        opt_params=opt_params, 
        output_dir=f"training_data/logs/{exp_folder}/{exp}/{name_str}" 
    )
    res, naeval, corr = uasp.evaluate_all(hsm, input_val_processed, output_val, output_val_mask)
    hsm.save_model(f"./training_data/models/{exp_folder}/{exp}/{name}__{i}.ndnmod")
with open("./trained_data/experiments.txt", "a+") as f:
    f.write(f"{exp_folder}/{exp}/{name}\n")



#
# 2
# Duplicate hidden layer with the same parameters.
#
name = 'baselineshort_exp32_1mult1e6_Dog16xN02xN02xN_LxSpxSpxSp_Nonorm_1e3x16x5000'
exp_folder = "experiments_1"
exp = "bs1_exp32"

def get_hsm_params_custom(input, output, i):
    _, output_shape = output.shape
    _, input_shape = input.shape
    pprint(f"in: {input_shape} out: {output_shape}")
        
    intput_w, input_h = int(math.sqrt(input_shape)), int(math.sqrt(input_shape))
    hsm_params = NDNutils.ffnetwork_params(
        verbose=False,
        input_dims=[1, intput_w, input_h], 
        layer_sizes=[16, int(0.2*output_shape),int(0.2*output_shape), output_shape], # paper: 9, 0.2*output_shape
        ei_layers=[None, None, None],
        normalization=[0, 0, 0, 0], 
        layer_types=['diff_of_gaussians', 'normal','normal','normal'],
        act_funcs=['lin', 'softplus', 'softplus','softplus'],
        reg_list={
            'd2x':[None,None,None,None],
            'l1':[None,None,None,None],
            'max':[None,None,None,None]})
    hsm_params['weights_initializers']=['random','normal','normal','normal']

    return hsm_params

def get_training_params():
    return {'batch_size': 16, 'use_gpu': False, 'epochs_summary': 100, 'epochs_training': 5000, 'learning_rate': 0.1e-3}

input_tr_processed, output_tr, output_tr_mask = udata.load_data_multiple(
    [1, 2], 'training', lambda x: x*1_000_000)
input_val_processed, output_val, output_val_mask = udata.load_data_multiple(
    [1, 2], 'validation', lambda x: x*1_000_000)

for i in range(10):
    seed = i
       
    hsm_params = get_hsm_params_custom(input_tr_processed, output_tr, i)
    pprint(hsm_params)
    hsm, input_tuple = unet.get_network(
        input_tr_processed, output_tr,
        'adam', 
        get_training_params(),
        hsm_params,
        'poisson',
        input_val_processed, output_val,
        output_tr_mask, output_val_mask,
        f"{name}__{i}", seed,
        
    )
    hsm.log_correlation = 'zero-NaNs'
        
    (input, output, train_indxs, test_indxs, data_filters, larg, opt_params, name_str) = input_tuple
    hsm.train(
        input_data=input, 
        output_data=output, 
        train_indxs=train_indxs, 
        test_indxs=test_indxs, 
        data_filters=data_filters,
        learning_alg=larg, 
        opt_params=opt_params, 
        output_dir=f"training_data/logs/{exp_folder}/{exp}/{name_str}" 
    )
    res, naeval, corr = uasp.evaluate_all(hsm, input_val_processed, output_val, output_val_mask)
    hsm.save_model(f"./training_data/models/{exp_folder}/{exp}/{name}__{i}.ndnmod")
with open("./trained_data/experiments.txt", "a+") as f:
    f.write(f"{exp_folder}/{exp}/{name}\n")


#
# 1
# Need more data on exp. 31
#
name = 'baselineshort_exp31_1mult1e6_Dog9xN0x2N_SpxSpxSp_Nonorm_1e3x16x5000'
exp_folder = "experiments_1"
exp = "bs1_exp31"

def get_hsm_params_custom(input, output, i):
    _, output_shape = output.shape
    _, input_shape = input.shape
    pprint(f"in: {input_shape} out: {output_shape}")
        
    intput_w, input_h = int(math.sqrt(input_shape)), int(math.sqrt(input_shape))
    hsm_params = NDNutils.ffnetwork_params(
        verbose=False,
        input_dims=[1, intput_w, input_h], 
        layer_sizes=[9, int(0.2*output_shape), output_shape], # paper: 9, 0.2*output_shape
        ei_layers=[None, None, None],
        normalization=[0, 0, 0], 
        layer_types=['diff_of_gaussians','normal','normal'],
        act_funcs=['softplus', 'softplus','softplus'],
        reg_list={
            'd2x':[None,None,None],
            'l1':[None,None,None],
            'max':[None,None,None]})
    hsm_params['weights_initializers']=['random','normal','normal']

    return hsm_params

def get_training_params():
    return {'batch_size': 16, 'use_gpu': False, 'epochs_summary': 100, 'epochs_training': 5000, 'learning_rate': 0.1e-3}

input_tr_processed, output_tr, output_tr_mask = udata.load_data_multiple(
    [1], 'training', lambda x: x*1_000_000)
input_val_processed, output_val, output_val_mask = udata.load_data_multiple(
    [1], 'validation', lambda x: x*1_000_000)

for i in range(10, 20):
    seed = i
       
    hsm_params = get_hsm_params_custom(input_tr_processed, output_tr, i)
    pprint(hsm_params)
    hsm, input_tuple = unet.get_network(
        input_tr_processed, output_tr,
        'adam', 
        get_training_params(),
        hsm_params,
        'poisson',
        input_val_processed, output_val,
        output_tr_mask, output_val_mask,
        f"{name}__{i}", seed,
        
    )
    hsm.log_correlation = 'zero-NaNs'
        
    (input, output, train_indxs, test_indxs, data_filters, larg, opt_params, name_str) = input_tuple
    hsm.train(
        input_data=input, 
        output_data=output, 
        train_indxs=train_indxs, 
        test_indxs=test_indxs, 
        data_filters=data_filters,
        learning_alg=larg, 
        opt_params=opt_params, 
        output_dir=f"training_data/logs/{exp_folder}/{exp}/{name_str}" 
    )
    res, naeval, corr = uasp.evaluate_all(hsm, input_val_processed, output_val, output_val_mask)
    hsm.save_model(f"./training_data/models/{exp_folder}/{exp}/{name}__{i}.ndnmod")
with open("./trained_data/experiments.txt", "a+") as f:
    f.write(f"{exp_folder}/{exp}/{name}\n")

