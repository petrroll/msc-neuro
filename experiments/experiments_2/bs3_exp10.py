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

import fire

def runner(exp_folder, exp, run, lin_scale, input_scale):
    run_1(exp_folder, exp, run, lin_scale, input_scale)

#
# based on bl3:
# - input normalization w/without lin_scale layer
# - with identity/b1-times-mil/std-mean normalization of input
def run_1(exp_folder, exp, run, lin_scale, input_scale):

    name = f'baseline3_ls{lin_scale}_iscale{input_scale}x5000'
    exp = f"{exp}x{run}"
    
    def get_hsm_params_custom(input, output, i):
        _, output_shape = output.shape
        _, input_shape = input.shape
        pprint(f"in: {input_shape} out: {output_shape}")

        intput_w, input_h = int(math.sqrt(input_shape)), int(math.sqrt(input_shape))
        hsm_params = NDNutils.ffnetwork_params(
            verbose=False,
            input_dims=[1, intput_w, input_h], 
            layer_sizes=([1] if lin_scale else []) + [9, int(0.2*output_shape), output_shape], # paper: 9, 0.2*output_shape
            ei_layers=([None] if lin_scale else []) + [None, None, None],
            normalization=([0] if lin_scale else []) + [0, 0, 0], 
            layer_types=(['lin_scale'] if lin_scale else []) + ['diff_of_gaussians','normal','normal'],
            act_funcs=(['lin'] if lin_scale else []) + ['lin', 'softplus','softplus'],
            reg_list={})
        hsm_params['weights_initializers']=([None] if lin_scale else [])['random','normal','normal']
        hsm_params['biases_initializers']=([None] if lin_scale else [])['trunc_normal','trunc_normal','trunc_normal']

        return hsm_params

    def get_training_params():
        epochs = 5000
        return {'batch_size': 16, 'use_gpu': False, 'epochs_summary': epochs//50, 'epochs_training': epochs, 'learning_rate': 0.001}

    if input_scale == "normalize_mean_std":
        input_proc = udata.normalize_mean_std
    elif input_scale == "times_mil":
        input_proc = lambda x: x*1_000_000
    else:
        input_proc = lambda x: x

    input_tr_processed, output_tr, output_tr_mask = udata.load_data_multiple(
        [1], 'training', input_proc)
    input_val_processed, output_val, output_val_mask = udata.load_data_multiple(
        [1], 'validation', input_proc)

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
    with open("./training_data/experiments.txt", "a+") as f:
        f.write(f"{exp_folder}/{exp}/{name}\n")

if __name__ == "__main__":
    fire.Fire(runner)
