import numpy as np 
import tensorflow as tf

import NDN3.NDNutils as NDNutils
import NDN3.NDN as NDN

from datetime import datetime

def get_network_inputs(train_input, train_output,
                        larg, opt_params,
                        test_input = None, test_output = None,
                        train_data_filters=None, test_data_filters=None, custom_name = None):
    '''
    Prepares inputs for NDN model instance.

    - Concatenates train a test data to create input_data, output_data, train_indxs, and test_indxs.
    - Prepares run name.

    Note:
        The API of this method is this convoluted for backwards compatibility reasons.
    '''

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

    return (input, output, train_indxs, test_indxs, data_filters, larg, opt_params, name_str)


def get_network(train_input, train_output, 
                  larg, opt_params, hsm_params, noise_dist='poisson',
                  test_input = None, test_output = None, 
                  train_data_filters=None, test_data_filters=None,
                  custom_name = None, 
                  seed=0):
    '''
    Prepares inputs for NDN model instance and then creates it. Returns both inputs and the new model instance.

    Essentially just a `get_network_inputs(...)` and creation of new NDN instance.

    Note:
        The API of this method is this convoluted for backwards compatibility reasons.

    '''
    # The seeds within NDN are applied only on ._build_graph which happens after weights get initialized
    # ..need to set it now -> when NDN gets created -> weight get initiated as part of ._define_network
    # Update: Since PR #18 on NDN this is not necessary, the seeds are properly appied on model creation.
    # .. To make sure old scripts are still 100 % reproducible I left it here as they might expect the seeds 
    # .. to be set here and not just when a new model instance is created.
    np.random.seed(seed)
    tf.set_random_seed(seed)
        
    input_params = get_network_inputs(train_input, train_output, 
                                        larg, opt_params, 
                                        test_input, test_output, 
                                        train_data_filters, test_data_filters, 
                                        custom_name)

    hsm = NDN.NDN(hsm_params, noise_dist=noise_dist, tf_seed=seed)
    return hsm, input_params

def train_network(train_input, train_output, 
                  larg, opt_params, hsm_params, noise_dist='poisson',
                  test_input = None, test_output = None, 
                  train_data_filters=None, test_data_filters=None,
                  custom_name = None, seed=0, logdir='logs'):
    '''
    Prepares inputs for NDN model instance, creates it, and then trains it. Returns trained model.

    Essentially just a `get_network(...)` and subsequent call to its train(...) method.

    Note:
        The API of this method is this convoluted for backwards compatibility reasons.

    '''

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
        output_dir=f"training_data/logs/{logdir}/{name_str}"
    )
    return hsm

def load_network(path, seed=None):
    '''
    Loads and returns a NDN model instance.
    '''
    if seed is not None:
        np.random.seed(seed)
        tf.set_random_seed(seed)

    return NDN.NDN.load_model(path)
