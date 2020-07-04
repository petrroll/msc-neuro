import numpy as np 
import tensorflow as tf

import NDN3.NDNutils as NDNutils
import NDN3.NDN as NDN

from datetime import datetime

SEED = 0
def get_network(train_input, train_output, 
                  larg, opt_params, hsm_params, noise_dist='poisson',
                  test_input = None, test_output = None, 
                  train_data_filters=None, test_data_filters=None,
                  custom_name = None, 
                  seed=0):

    # The seeds within NDN are applied only on ._build_graph which happens after weights get initialized
    # ..need to set it now -> when NDN gets created -> weight get initiated as part of ._define_network
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    time_str = datetime.now().strftime("%d-%m_%H-%M-%S")
    name_str = time_str + "_" + custom_name if custom_name is not None else time_str
    
    train_len, _ = train_input.shape
    if test_input is not None:
        input = np.concatenate([train_input, test_input], axis=0)
        output = np.concatenate([train_output, test_output], axis=0)
        data_filters = np.concatenate([train_data_filters, test_data_filters], axis=0) if train_data_filters is not None else None
        test_len, _ = test_input.shape
    else:
        input = train_input
        output = train_output
        data_filters = train_data_filters
        opt_params['early_stop'] = 0 # If we don't have test data -> shouldn't be early stopping (could early stop on train)
        test_len = 0  
        
    train_indxs = np.array(range(train_len))
    test_indxs = np.array(range(train_len, train_len + test_len)) if test_len > 0 else None
    
    hsm = NDN.NDN(hsm_params, noise_dist=noise_dist, tf_seed=seed)
    return hsm, (input, output, train_indxs, test_indxs, data_filters, larg, opt_params, name_str)

def train_network(train_input, train_output, 
                  larg, opt_params, hsm_params, noise_dist='poisson',
                  test_input = None, test_output = None, 
                  train_data_filters=None, test_data_filters=None,
                  custom_name = None, seed=0, logdir='logs'):
    hsm, (input, output, train_indxs, test_indxs, data_filters, larg, opt_params, name_str) = get_network(
                  train_input, train_output, 
                  larg, opt_params, hsm_params, noise_dist,
                  test_input, test_output, 
                  train_data_filters, test_data_filters,
                  custom_name, seed)
    hsm.train(
        input_data=input, 
        output_data=output, 
        train_indxs=train_indxs, 
        test_indxs=test_indxs, 
        data_filters=data_filters,
        learning_alg=larg, 
        opt_params=opt_params, 
        output_dir=f"logs/{logdir}/{name_str}"
    )
    return hsm

def load_network(path, seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    return NDN.NDN.load_model(path)
