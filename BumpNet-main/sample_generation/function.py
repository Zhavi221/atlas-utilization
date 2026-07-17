import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path
import argparse
import yaml
    
sys.path.append(str(Path(__file__).resolve().parent))

import function_backgrounds

def get_linear_bin_edges(xmin, xmax, nbins):
    """
    Return an array of bin edges spaced linearly between [xmin, xmax], where
    nbins is the resulting number of bins.

    Parameters
    ----------
    xmin, xmax : float
        min and max edge values
    nbins : int
        number of bins

    Returns
    -------
    bin_edges : array
        array of bin edge values
    """
    return np.linspace(xmin, xmax, num=nbins+1)


def get_linear_bin_centers(xmin, xmax, nbins):
    """
    Return an array of bin centers spaced linearly between (xmin, xmax). Does
    not including these edge values.

    See : get_linear_bin_edges(xmin, xmax, nbins)

    Parameters
    ----------
    xmin, xmax : float
        min and max edge values
    nbins : int
        number of bins

    Returns
    -------
    bin_centers : array
        array of bin center values
    """
    bin_edges = get_linear_bin_edges(xmin, xmax, nbins)
    return (bin_edges[1:] + bin_edges[:-1]) / 2


def function(config):

    # Import smoothed curves
    if "smooth_dir" in config:
        smooth_dir = config["smooth_dir"]
        bin_content_smooth = np.load(f'{smooth_dir}/background.npy', allow_pickle=True)
        bin_edges_smooth = np.load(f'{smooth_dir}/bin_edges.npy', allow_pickle=True)

        print(f'Loaded {len(bin_content_smooth)} smoothed histograms from {smooth_dir}')

        # Determine dynamic range of histograms
        min_nbins = max(min([len(content) for content in bin_content_smooth]), 25) # enforce a minimum of 25 bins
        max_nbins = max([len(content) for content in bin_content_smooth])
        bins_range = (min_nbins, max_nbins)
        xmin = round(min([min(edges) for edges in bin_edges_smooth]))
        xmax = round(max([max(edges) for edges in bin_edges_smooth]))
        min_ymin = round(max(min([min(content) for content in bin_content_smooth]), 0.1), 1) # enforce a minimum of 0.1 for y_min
        max_ymin = round(max([min(content) for content in bin_content_smooth]))
        min_ymax = round(min([max(content) for content in bin_content_smooth]))
        max_ymax = round(max([max(content) for content in bin_content_smooth]))

        if max_ymin > min_ymax: min_ymax = max_ymin + 1 # ensure that min_ymax > max_ymin

    elif "custom_parameters" in config:
        print("No smoothed directory provided. Using custom dynamic range.")
        custom_params = config["custom_parameters"]
        bins_range, xmin, xmax, min_ymin, max_ymin, min_ymax, max_ymax = (
            custom_params['nbins'],
            custom_params['min_x'], custom_params['max_x'],
            custom_params['min_ymin'], custom_params['max_ymin'],
            custom_params['min_ymax'], custom_params['max_ymax']
        )

    else:
        print("Warning: No smoothed directory provided. Using default dynamic range.")
        bins_range = (30, 100)
        xmin, xmax, min_ymin, max_ymin, min_ymax, max_ymax = 1, 1000, 0.1, 3, 10, 1000000

    number_of_samples = config['number_of_samples']
    
    # Initialize function generation
    background_functions = config["background_functions"]
    seed = config["seed"]
    rng = np.random.default_rng(seed)
    samples_per_function = int(np.ceil(number_of_samples / len(background_functions)))

    bkgs = []
    names = []
    binning = []

    output_dir = config["output_dir"]

    # Sample random numbers between 0.01 and 10 (Arbitrary choice) for the exponent on the function
    if 'one_over_x_to_nth' in background_functions:
        rnd_n = rng.uniform(0.01, 10, size=samples_per_function).astype('longdouble') #Randomly generate n values for the newest incorporated function.

    rnd_ymins = rng.uniform(min_ymin, max_ymin, size=(len(background_functions)* samples_per_function)).astype('longdouble')
    rnd_ymaxs = rng.uniform(min_ymax, max_ymax, size=(len(background_functions)* samples_per_function)).astype('longdouble')

    y_mask = rnd_ymins < rnd_ymaxs

    rnd_ymins = rnd_ymins[y_mask]
    rnd_ymaxs = rnd_ymaxs[y_mask]

    amount_diff = len(background_functions)*samples_per_function - y_mask.sum()

    while amount_diff > 0:
        rnd_ymins = np.concatenate((rnd_ymins,rng.uniform(min_ymin, max_ymin, size=amount_diff).astype('longdouble')))
        rnd_ymaxs = np.concatenate((rnd_ymaxs,rng.uniform(min_ymax, max_ymax, size=amount_diff).astype('longdouble')))

        y_mask = rnd_ymins < rnd_ymaxs
        
        rnd_ymins = rnd_ymins[y_mask]
        rnd_ymaxs = rnd_ymaxs[y_mask]

        amount_diff = len(background_functions)*samples_per_function - y_mask.sum()
    
    rnd_ymins = rnd_ymins.reshape(len(background_functions), samples_per_function)
    rnd_ymaxs = rnd_ymaxs.reshape(len(background_functions), samples_per_function)

    rnd_x_float_pairs = rng.uniform(xmin, xmax,
                                           size=(len(background_functions), samples_per_function, 2)).astype('longdouble')
    rnd_x_float_pairs.sort(axis=-1)
    rnd_xmins, rnd_xmaxs = rnd_x_float_pairs[:, :, 0], rnd_x_float_pairs[:, :, 1]

    # Generate the bin centers for each pair of randomly generated x_min, x_max.


    for i, func in enumerate(background_functions):
        print(f'Generating bkg samples for {func}')
        # Get bkg methods
        f = getattr(function_backgrounds, func)
        f_params = getattr(function_backgrounds, func+'_params')
        for j in tqdm(range(samples_per_function)):
            # Get bkg distributions

            # Get a random bin number and generate bin centers
            if len(bins_range) > 1:
                nbins = rng.integers(low=bins_range[0], high=bins_range[1])
                max_nb_of_bins = bins_range[1]
            else:
                nbins = bins_range[0]
                max_nb_of_bins = bins_range[0]

            bin_edges = get_linear_bin_edges(rnd_xmins.T[j][i], rnd_xmaxs.T[j][i], nbins)
            bin_centers = get_linear_bin_centers(rnd_xmins.T[j][i], rnd_xmaxs.T[j][i], nbins).T

            if func == 'one_over_x_to_nth': 
                p0, p1 = f_params(bin_centers[0], rnd_ymaxs[i][j], bin_centers[-1], rnd_ymins[i][j], rnd_n[j])
                bkg = f(bin_centers, p0, p1, rnd_n[j]).astype(float)

            else:
                p0, p1 = f_params(bin_centers[0], rnd_ymaxs[i][j], bin_centers[-1], rnd_ymins[i][j])
                if func == 'parabola_half':
                    bkg = f(bin_centers, p0, p1, bin_centers[-1], rnd_ymins[i][j]).astype(float)
                elif func == 'cos_quarter':
                    bkg = f(bin_centers, p0, p1, rnd_ymaxs[i][j], rnd_ymins[i][j]).astype(float)
                elif func == 'cosh_half':
                    bkg = f(bin_centers, p0, p1, bin_centers[-1]).astype(float)
                else:
                    bkg = f(bin_centers, p0, p1).astype(float)

            if bkg[np.isfinite(bkg)].shape[0] == 0:
                print(f'{func} too big: skipping')
                continue
            
            if config["rescale_minimum"]!=0:
                # Scale mini to RescaleMininmum, but keep max as it is, log variation
                bkg = bkg.astype(float)
                Minimum = np.min(bkg)
                Minimum_i = np.argmin(bkg)
                Maximum = np.max(bkg)
                Maximum_i = np.argmax(bkg)
                if Minimum<config["rescale_minimum"]: Minimum=config["rescale_minimum"]
                bkg *= np.exp(np.log(config["rescale_minimum"]/Minimum)*(bin_centers-bin_centers[Maximum_i])/(bin_centers[Minimum_i]-bin_centers[Maximum_i]))

            if func == 'one_over_x_to_nth':
                name = lambda precision: f'{func}_{float(f"{p0:.{precision}g}"):g}_{float(f"{p1:.{precision}g}"):g}_{float(f"{rnd_n[j]:.{precision}g}"):g}'
            else:
                name = lambda precision: f'{func}_{float(f"{p0:.{precision}g}"):g}_{float(f"{p1:.{precision}g}"):g}'

            significant_digits = 2  # Default to 2 significant digits for parameters
            while [name(significant_digits)] in names: 
                significant_digits += 1
                if significant_digits > 10:
                    print('Could not generate unique name for function after 10 significant digits. Skipping.')
                    break

            names.append([name(significant_digits)])
            binning.append(bin_edges)
            bkgs.append(bkg)

    Path(f'{output_dir}').mkdir(parents=True, exist_ok=True)

    # copy config to output dir
    config_file = f'{output_dir}/config.yaml'
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    for label, data in zip(['names', 'background', 'bin_edges'], [names, bkgs, binning]):
        out_file = f'{output_dir}/{label}.npy'
        np.save(out_file, np.array(data, dtype=object))
        print(f'> Saved {len(data)} histograms to {out_file}')

    return f"{output_dir}"

def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--seed', help=' random seed')
    arg_parse.add_argument('--output_dir', help='output directory for generated functions')
    arg_parse.add_argument('--smooth_dir', help='input directory of smoothed histograms for dynamic range')
    
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = config["function"]
    
    if args.seed is not None:
        config["seed"] = args.seed
    
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    
    if args.smooth_dir is not None:
        config["smooth_dir"] = args.smooth_dir


    print('>>>>> Starting generating funtions step...')

    function(config)

    print('>>>>> Finished generating functions step.')

if __name__ == '__main__':
    main()
