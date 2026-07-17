import numpy as np

def get_1d_hist(x, xmin=None, xmax=None, density=False, binning=None):

    if binning is not None:
        bins = binning
    else:
        bins = np.arange(x.min(), x.max(), 0.1)

    entries, bin_edges = np.histogram(x, bins=bins, density=density)
    bin_widths = np.array([bin_edges[i+1]-bin_edges[i] for i in range(0, len(bin_edges)-1)])
    bin_centers = bin_edges[:-1] + bin_widths/2

    return entries, bin_centers
