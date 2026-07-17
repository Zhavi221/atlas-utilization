import logging
import numpy as np
from scipy.interpolate import PchipInterpolator
import logging
import os
from pathlib import Path
from tqdm import tqdm


def stretch(config):
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
    output_func = []

    logging.debug(f'number of samples: {n_samples}')
    logging.debug(f'number of bins: {n_bins}')

    rng = np.random.default_rng(seed=config["seed"])

    print("stretching each sample")
    for i in tqdm(range(n_samples)):
        current_bins = bins_data[i][~np.isnan(bins_data[i])]
        current_func = func_data[i][~np.isnan(func_data[i])]

        logging.debug(f'bin edges: {current_bins}')
        logging.debug(f'number of bin edges: {current_bins.shape[0]}')
        logging.debug(f'number of func values: {current_func.shape[0]}')

        current_centers = (current_bins[1:] + current_bins[:-1])/2
        current_sizes = current_bins[1:] - current_bins[:-1]

        logging.debug(f'number of bin centers: {current_centers.shape[0]}')

        if config["stretch_direction"] == "right" or config["stretch_direction"] == "left":
            logging.debug("choosing a random index to stretch from. Since we will refit with the same function, we should have at least 5 points of data. So, n>=5.")
            n_rnd = rng.integers(low=5,high=len(current_sizes)+1)
            logging.debug(f'selected random int: {n_rnd}')
        
            if config["stretch_direction"] == "right":
                relative_sizes = current_sizes[:n_rnd] / current_sizes[:n_rnd].sum()
                new_func = current_func[:n_rnd]

            if config["stretch_direction"] == "left":
                relative_sizes = current_sizes[-n_rnd:] / current_sizes[-n_rnd:].sum()
                new_func = current_func[-n_rnd:]

        if config["stretch_direction"] == "both":
            logging.debug(f'choosing two random ints with difference 5.')
            while True:
                n2_rnd = rng.integers(low=0, high=len(current_sizes)+1,size=2)
                n2_rnd.sort()
                if n2_rnd[1]-n2_rnd[0] >= 5:
                    break
            logging.debug(f'selected random ints ({n2_rnd[0]},{n2_rnd[1]})')

            relative_sizes = current_sizes[n2_rnd[0]:n2_rnd[1]] / current_sizes[n2_rnd[0]:n2_rnd[1]].sum()
            new_func = current_func[n2_rnd[0]:n2_rnd[1]]

        new_sizes = current_sizes.sum() * relative_sizes

        new_bins = np.array([])
        for j in np.arange(len(new_sizes)+1):
            new_bins = np.append(new_bins,current_bins[0] + new_sizes[:j].sum())
        logging.debug(f'new bin edges: {new_bins}')
        logging.debug(f'there are now {new_bins.shape[0]} edges')
        
        new_centers = (new_bins[1:] + new_bins[:-1]) / 2

        logging.debug(f'number of new bin centers: {new_centers.shape[0]}')
        logging.debug(f'number of new func: {new_func.shape[0]}')
        
        interp = PchipInterpolator(new_centers,new_func)
        new_fit = interp(current_centers)

        logging.debug('finished interpolating, concatenating and appending for output now')

        # output_fit = np.concatenate((new_fit,np.ones(n_bins - len(new_fit))*np.nan))

        output_func.append(new_fit)

    output_func = np.array(output_func, dtype=object)

    logging.debug(f'type of output array: {output_func.dtype}')

    print('saving stretched func')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f'{output_dir}/{config["name"]}_func.npy', output_func, allow_pickle=True)
    np.save(f'{output_dir}/{config["name"]}_bins.npy', bins_data, allow_pickle=True)
    np.save(f'{output_dir}/{config["name"]}_name.npy', name_data, allow_pickle=True)



    