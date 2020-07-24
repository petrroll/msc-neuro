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
import utils.misc as umisc

import fire

def add_normal_noise_and_shift_positive(input, scale):
    noised = input + np.random.normal(loc=0, scale=np.sqrt(input)*scale)    # Assuming variance (std^2) is proportional to magnitude ~ poisson assumption
    return noised + np.min(noised)                                          # Output should be positive (assumption in original data)

def runner(exp_folder, exp, run, noise_coef, eval_noised):
    run_1(exp_folder, exp, run, noise_coef, eval_noised)

#
# based on bl3:
# - generate output data through the best model we have so far (output_XX_gen)
# - noise it up with gaussian noise that has variance proportional to magnitude (~poisson assumption) (output_XX_gen_noise)
# - train a new network against the noised up data 
# - evaluate w.r.t to either noised validate (output_val_gen_noise) or purely generated (output_val_gen)
# -> helps us get some (!just some!) intuition around following questions:
#   - can the architecture learn when noised data is used?
#   - how much noise gets learned?
#   - how much underlying model (that got noised) gets learned?
def run_1(exp_folder, exp, run, noise_coef, eval_noised):
    np.random.seed(42)

    gen_model_path, = udata.get_file_paths_filtered("./training_data/models/baseline/bl3/", r"__14")
    gen_network = NDN.NDN.load_model(gen_model_path)

    input_tr_processed, output_tr, output_tr_mask = udata.load_data_multiple(
        [1], 'training', udata.normalize_mean_std)
    input_val_processed, output_val, output_val_mask = udata.load_data_multiple(
        [1], 'validation', udata.normalize_mean_std)

    output_tr_gen = gen_network.generate_prediction(input_tr_processed)
    output_val_gen = gen_network.generate_prediction(input_val_processed)

    import functools
    # Add noise to produce desired correlation between `output_tr_gen` (our assumed truth) & `output_tr_gen_noise` (what we'll train against) 
    # .. as is the correlation between `output_tr_gen` (best model) and `output_tr` (gold data)
    output_tr_gen_noise = add_normal_noise_and_shift_positive(output_tr_gen, noise_coef)
    # To generate noised [val]idation data we'll average a number of runs to get desired correlation (as above with [tr]aining data)
    val_avg_runs = 3
    output_val_gen_noise_runs = [add_normal_noise_and_shift_positive(output_val_gen, noise_coef) for i in range(val_avg_runs)]
    output_val_gen_noise = functools.reduce(np.add, output_val_gen_noise_runs, np.zeros_like(output_val_gen_noise_runs[0]))/val_avg_runs

    tr_noise_corr = np.mean(uas.get_correlation(output_tr_gen, output_tr_gen_noise))
    val_noise_corr = np.mean(uas.get_correlation(output_val_gen, output_val_gen_noise))

    name = f'baseline3_gen_bl3__14_noise{noise_coef}xValAvg{val_avg_runs}_trNoiseCorr{tr_noise_corr}xValNoiseCorr{val_noise_corr}_evalNoise{eval_noised}x5000'
    exp = f"{exp}x{run}"
    
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
            act_funcs=['lin', 'softplus','softplus'],
        )
        hsm_params['weights_initializers']=['random','normal','normal']
        hsm_params['biases_initializers']=['trunc_normal','trunc_normal','trunc_normal']

        return hsm_params

    def get_training_params():
        epochs = 5000
        return {'batch_size': 16, 'use_gpu': False, 'epochs_summary': epochs//50, 'epochs_training': epochs, 'learning_rate': 0.001}

    input_tr_processed, output_tr, output_tr_mask = udata.load_data_multiple(
        [1], 'training', udata.normalize_mean_std)
    input_val_processed, output_val, output_val_mask = udata.load_data_multiple(
        [1], 'validation', udata.normalize_mean_std)

    # Trained with generated noised data instead of just generated data (true gold data) 
    # Evaluated either on true generated data (gold) or noised
    output_tr_used = output_tr_gen_noise
    output_val_used = output_val_gen_noise if eval_noised else output_val_gen

    for i in range(10):
        seed = i

        hsm_params = get_hsm_params_custom(input_tr_processed, output_tr_used, i)
        pprint(hsm_params)
        hsm, input_tuple = unet.get_network(
            input_tr_processed, output_tr_used,
            'adam', 
            get_training_params(),
            hsm_params,
            'poisson',
            input_val_processed, output_val_used,
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
        with umisc.HiddenPrints():
            _, _, corr_val_gen_noised = uasp.evaluate_all(hsm, input_val_processed, output_val_gen_noise, output_val_mask)
            _, _, corr_val_gen = uasp.evaluate_all(hsm, input_val_processed, output_val_gen, output_val_mask)
            _, _, corr_val_true = uasp.evaluate_all(hsm, input_val_processed, output_val, output_val_mask)

            _, _, corr_tr_gen_noised = uasp.evaluate_all(hsm, input_tr_processed, output_tr_gen_noise, output_tr_mask)
            _, _, corr_tr_gen = uasp.evaluate_all(hsm, input_tr_processed, output_tr_gen, output_tr_mask)
            _, _, corr_tr_true = uasp.evaluate_all(hsm, input_tr_processed, output_tr, output_tr_mask)

        print(f"corr_val_gen_noised{corr_val_gen_noised} corr_val_gen{corr_val_gen} corr_val_true{corr_val_true}")
        print(f"corr_tr_gen_noised{corr_tr_gen_noised} corr_tr_gen{corr_tr_gen} corr_tr_true{corr_tr_true}")

        hsm.save_model(f"./training_data/models/{exp_folder}/{exp}/{name}__{i}.ndnmod")
    with open("./training_data/experiments.txt", "a+") as f:
        f.write(f"{exp_folder}/{exp}/{name}\n")

if __name__ == "__main__":
    fire.Fire(runner)
