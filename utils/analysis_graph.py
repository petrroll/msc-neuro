import numpy as np
import PIL

from IPython.display import display

def reshape_single_as_picture(input, size):
    return np.reshape(input, size)

def as_single_picture(input, size, outsize=None):
    outsize = outsize if outsize is not None else size 
    input_as_uint8 = input.astype(np.uint8)
    return PIL.Image.fromarray(reshape_single_as_picture(input_as_uint8, size), 'L').resize(outsize, PIL.Image.NEAREST)

import seaborn as sns
import matplotlib.pyplot as plt
def display_as_single_heatmap(input, size):
    display(sns.heatmap(np.reshape(input, size), center=0.0))
    plt.show()

def plot_square_w_as_heatmaps(w, get_data, plots=(3, 3)):
    nrows, ncols = plots
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(16, 16)
    for i in range(min(nrows*ncols, w.shape[1])):
        dta = get_data(w, i).flatten()
        size = int(np.sqrt(dta.shape[0]))
        dta = np.reshape(dta, (size, size))
        sns.heatmap(dta, center=0.0, ax=ax[i//ncols][i%ncols])

    plt.show()
