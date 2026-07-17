import logging
import numpy as np
from scipy.interpolate import PchipInterpolator
import logging
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def systematics(config):
    if config["verbose"]:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    output_dir = f'{config["output_dir"]}/{config["name"]}'
    
    # Load all backgrounds
    bkg = []
    bin_edges = []
    names = []
    for bkg_type, bkg_files in config["backgrounds"].items():

        # Check if the input files are .npy files
        for path in bkg_files.values():
            if not path.lower().endswith(".npy"):
                raise ValueError(f"The input must be a .npy file, instead got: {path}")

        # Load background histograms from file
        bkg_i = np.load(bkg_files['hist'], allow_pickle=True)
        bin_edges_i = np.load(bkg_files['binning'], allow_pickle=True)
        name_i = np.load(bkg_files['name'], allow_pickle=True)

        nbins_max_i = max(len(hist) for hist in bkg_i)
        print(f'Loaded {bkg_type} backgrounds with {len(bkg_i)} histograms and nbins_max = {nbins_max_i}')

        # Combine with other backgrounds
        bkg.append(bkg_i)
        bin_edges.append(bin_edges_i)
        names.append(name_i)

    func_data = np.concatenate(bkg, axis=0)
    bins_data = np.concatenate(bin_edges, axis=0)
    name_data = np.concatenate(names, axis=0)

    logging.debug(f'type of bins arr: {bins_data.dtype}')
    logging.debug(f'type of func arr: {func_data.dtype}')
    logging.debug(f'type of name arr: {name_data.dtype}')
    
    n_samples = len(bins_data)
    n_bins = max(len(edges) for edges in bins_data) - 1
    standard_bias_func = []
    reverse_bias_func = []
    inferior_bias_func = []
    superior_bias_func = []

    logging.debug(f'number of samples: {n_samples}')
    logging.debug(f'number of bins: {n_bins}')
    
    print("applying systematics over each sample")
    for i in tqdm(range(bins_data.shape[0])):
        bkg_i = func_data[i][~np.isnan(func_data[i])]
        logging.debug(f'np.nan in bkg: {np.any(np.isnan(bkg_i))}')
        x = np.arange(len(bkg_i))
        
        standard_bias_func.append(1.5 * ((len(x) - x)/len(x))*bkg_i)
        reverse_bias_func.append( 0.5 * ((len(x) - x) / len(x)) * bkg_i)
        inferior_bias_func.append( 0.5 * bkg_i)
        superior_bias_func.append( 1.5 * bkg_i)

    standard_func = np.array(standard_bias_func, dtype=object)
    reverse_func = np.array(reverse_bias_func, dtype=object)
    inferior_func = np.array(inferior_bias_func, dtype=object)
    superior_func = np.array(superior_bias_func, dtype=object)

    logging.debug(f'type of output array: {standard_func.dtype}')

    print('saving systematics func')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f'{output_dir}/{config["name"]}_standard_func.npy', standard_func, allow_pickle=True)
    np.save(f'{output_dir}/{config["name"]}_reverse_func.npy', reverse_func, allow_pickle=True)
    np.save(f'{output_dir}/{config["name"]}_inferior_func.npy', inferior_func, allow_pickle=True)
    np.save(f'{output_dir}/{config["name"]}_superior_func.npy', superior_func, allow_pickle=True)
    np.save(f'{output_dir}/{config["name"]}_bins.npy', bins_data, allow_pickle=True)
    np.save(f'{output_dir}/{config["name"]}_name.npy', name_data, allow_pickle=True)