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

def runner(exp_folder, exp, run, cd2x, norm2_filt, l1, c_filters):
    run_1(exp_folder, exp, run, cd2x, norm2_filt, l1, c_filters)

#
# based on bs4: based on what/where paper (6942 Neural system identification for large populations separating “what” and “where”
def run_1(exp_folder, exp, run, cd2x, norm2_filt, l1, c_filters):
    name = f'baseline4_whatwhere_Cd2x{cd2x}xnorm2{norm2_filt}_filtxl1{l1}_convsize{c_filters}_x5000'
    exp = f"{exp}x{run}"

    def get_hsm_params_custom(input, output, i):
        _, output_shape = output.shape
        _, input_shape = input.shape
        pprint(f"in: {input_shape} out: {output_shape}")

        intput_w, input_h = int(math.sqrt(input_shape)), int(math.sqrt(input_shape))
        hsm_params = NDNutils.ffnetwork_params(
            verbose=False,
            input_dims=[1, intput_w, input_h], 
            layer_sizes=[c_filters, c_filters, c_filters, output_shape], # paper: 9, 0.2*output_shape
            ei_layers=[None, None, None, None],
            normalization=[0, 0, 0, 0], 
            layer_types=['conv','conv','conv','sep'],
            act_funcs=['softplus', 'softplus', 'softplus','softplus'],
            
            shift_spacing=[2,2,2,0],
            conv_filter_widths=[13,3,3, 0],

            reg_list={
                'd2x': [cd2x, cd2x, cd2x, None],
                'norm2_filt': [norm2_filt, norm2_filt, norm2_filt, None],
                'l1': [None, None, None, l1]
                })
        hsm_params['weights_initializers']=['normal','normal','normal', 'normal']
        hsm_params['biases_initializers']=['trunc_normal','trunc_normal','trunc_normal','trunc_normal']

        return hsm_params

    def get_training_params():
        epochs = 5000
        return {'batch_size': 16, 'use_gpu': False, 'epochs_summary': epochs//50, 'epochs_training': epochs, 'learning_rate': 0.001}

    input_tr_processed, output_tr, output_tr_mask = udata.load_data_multiple(
        [1], 'training', udata.normalize_mean_std)
    input_val_processed, output_val, output_val_mask = udata.load_data_multiple(
        [1], 'validation', udata.normalize_mean_std)

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
