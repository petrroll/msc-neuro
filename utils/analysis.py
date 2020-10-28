import numpy as np

def get_correlation(a, b, mask=None):
    '''
    Computes pearson R coefficient across first dimension of two 2-dimensional arrays (output_dim, data_num).

    Zeroes NaN (inc. +-inf) correlations. Supports mask (output_dim, data_num) with the same semantics as data_filters in NDN3.
    '''
    import scipy.stats
    assert a.shape == b.shape
    
    mask = mask if mask is not None else np.ones(a.shape)
    c = np.array([
        scipy.stats.pearsonr(
            a[mask[:, i]==1,i], 
            b[mask[:, i]==1,i]
        )[0] if np.sum(mask[:, i]) > 0 else 0
        for i 
        in range(a.shape[1])
    ])
    
    return np.nan_to_num(c, copy=False, nan=0.0)

def get_gaussian(weights, size, impl_sigma=True, impl_concentric=True):
    '''
    Gets fully realized gaussian filter based on DiffOfGaussiansLayer layer weights. 

    - size: size of the gaussian filter.
    - impl_sigma: whether the sigmas should be relative (True for current implementation).
    - impl_concentric: force co-centric Gaussians (True for current implementation)
    '''
    def get_single_gauss(alpha, sigma, ux, uy):
        return (alpha) * (np.exp(-((X - ux) ** 2 + (Y - uy) ** 2) / 2 / sigma) / (2*sigma*np.pi))
    
    xSize, ySize = size

    xCoords = np.array(range(xSize))
    yCoords = np.array(range(ySize))

    X, Y = np.meshgrid(xCoords, yCoords)
    X = np.expand_dims(X, 2)
    Y = np.expand_dims(Y, 2)

    alpha, sigma, ux, uy = weights[:4]
    res = get_single_gauss(alpha, sigma, ux, uy)

    alpha, sigma, ux, uy = weights[4:] 
    if impl_concentric:
        ux, uy = weights[2:4]      

    if impl_sigma:
        sigma += weights[1] 
    res -= get_single_gauss(alpha, sigma, ux, uy)

    return res