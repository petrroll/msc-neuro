import numpy as np
import NDN3.NDNutils as NDNutils
import NDN3.NDN as NDN
from datetime import datetime

inp = np.ones((128, 15*15))
out = np.ones((128, 20))

def get_hsm_params(input, output, hls=40):
    _, output_shape = output.shape

    d2x = 0.0005
    l1 = 0.000001

    hsm_params = NDNutils.ffnetwork_params(
        verbose=False,
        input_dims=[1, 15, 15], 
        layer_sizes=[5, 4, 20, output_shape],
        ei_layers=[0, 0],
        normalization=[0], 
        layer_types=['conv','diff_of_gaussians', 'normal','normal'],
        shift_spacing=[2, 0, 0, 0],
        conv_filter_widths=[3, 0, 0, 0],
        reg_list={
            'd2x':[d2x,None,None],
            'l1':[l1,None,None],
            'max':[None,None,100]})
    hsm_params['weights_initializers']=['normal','random','normal', 'normal']
    hsm_params['normalize_weights']=[1,0,0, 0]
    
    return hsm_params

def train_network(train_input, train_output, 
                  larg, opt_params, hsm_params, 
                  test_input = None, test_output = None, 
                  train_data_filters=None, test_data_filters=None):
    time_str = "test-" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    
    train_len, _ = train_input.shape

    input = train_input
    output = train_output
    data_filters = train_data_filters
    opt_params['early_stop'] = 0 # If we don't have test data -> shoudn't be early stopping (could early stop on train)
    test_len = 0  
        
    train_indxs = np.array(range(train_len))
    test_indxs = np.array(range(train_len, train_len + test_len)) if test_len > 0 else None
    
    hsm = NDN.NDN(hsm_params, noise_dist='poisson')
    hsm.train(
        input_data=input, 
        output_data=output, 
        train_indxs=train_indxs, 
        test_indxs=test_indxs, 
        data_filters=data_filters,
        learning_alg=larg, 
        opt_params=opt_params, 
        output_dir=f"logs/{time_str}"
    )
    
    return hsm


hsm_params = get_hsm_params(inp, out)
hsm = train_network(
    inp, out,
    'adam', 
    {'batch_size': 47, 'use_gpu': False, 'epochs_summary': 10, 'epochs_training': 11, 'learning_rate': 0.5e-4},
    hsm_params,
    )


