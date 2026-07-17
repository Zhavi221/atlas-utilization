import numpy as np
import pandas as pd
import argparse
import yaml
import os
import sys

from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from copy import deepcopy

import signals
from workspace import Workspace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utilities.data_loading import load_ATLAS



def calc_z(parameters):
    """Calculates the significance across a histogram initialized in Workspace."""
    i, wksp = parameters

    # Initialize signal histogram
    wksp.bin_widths = np.diff(wksp.bin_edges)
    wksp.bin_centers = wksp.bin_edges[:-1] + wksp.bin_widths/2

    # Scan over M in bin centers and test significance of signal centered there
    z_scan = wksp.z_scan()

    return z_scan


def load_smooth(dir_path): 
    """Loads smoothed histograms from .npy files located in dir_path."""
    if not os.path.exists(dir_path): raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    background = np.load(f"{dir_path}/background.npy", allow_pickle=True)
    bin_content = np.load(f"{dir_path}/bin_content.npy", allow_pickle=True)
    bin_edges = np.load(f"{dir_path}/bin_edges.npy", allow_pickle=True)
    names = np.load(f"{dir_path}/names.npy", allow_pickle=True)

    data = {
        'background': background,
        'bin_content': bin_content,
        'bin_edges': bin_edges,
        'names': names,
    }

    print(f'> Successfully loaded {len(data["background"])} histograms from {dir_path}')
    return data


def prepare_bsm(config):

    input_path = config["input_path"]
    output_dir = config["output_dir"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "smooth_dir" in config.keys():
        smooth_dir = config["smooth_dir"]

        print(f'Preparing BSM sample located in {input_path} using smoothed backgrounds from {smooth_dir}.')

        bsm_data = load_ATLAS(input_path, cuts=config, MC=False)
        smooth_data = load_smooth(smooth_dir)

        filtered_data = {
            'bin_content': [],
            'bin_content_bsm': [],
            'background': [],
            'bin_edges': [],
            'bin_errors': [],
            'names': [],
        }

        for i, name in enumerate(tqdm(smooth_data['names'], desc='Matching histograms', total=len(smooth_data['names']))):

            # Match BSM histograms to smoothed backgrounds by name
            mask = np.char.find(bsm_data['names'].astype(str), str(name[0])) >= 0
            match_idx = np.where(mask)[0]

            if len(match_idx) == 0:
                print(f'No match found for {name}, skipping.')
                continue
            if len(match_idx) > 1:
                print(f'Warning: Multiple matches found for {name}, skipping.')
                continue
            
            match_idx = match_idx[0]

            nbins = len(smooth_data['background'][i])
            
            # Ensure that bin edges are the same between SM and BSM
            smooth_bin_edges = np.round([smooth_data['bin_edges'][i][0], smooth_data['bin_edges'][i][-1]], 2)
            bsm_bin_edges = np.round([bsm_data['bin_edges'][match_idx][0], bsm_data['bin_edges'][match_idx][-1]], 2)
            bin_edges = [np.max([smooth_bin_edges[0], bsm_bin_edges[0]]), np.min([smooth_bin_edges[1], bsm_bin_edges[1]])]

            smooth_start_bin = np.where(np.round(smooth_data['bin_edges'][i], 2) == bin_edges[0])[0][0]
            smooth_end_bin = np.where(np.round(smooth_data['bin_edges'][i], 2) == bin_edges[1])[0][0]
            bsm_start_bin = np.where(np.round(bsm_data['bin_edges'][match_idx], 2) == bin_edges[0])[0][0]
            bsm_end_bin = np.where(np.round(bsm_data['bin_edges'][match_idx], 2) == bin_edges[1])[0][0]

            filtered_data['bin_content'].append(bsm_data['bin_content'][match_idx][bsm_start_bin:bsm_end_bin])
            filtered_data['bin_content_bsm'].append(bsm_data['bin_content_bsm'][match_idx][bsm_start_bin:bsm_end_bin])
            filtered_data['bin_edges'].append(bsm_data['bin_edges'][match_idx][bsm_start_bin:bsm_end_bin+1])
            filtered_data['bin_errors'].append(bsm_data['bin_errors'][match_idx][bsm_start_bin:bsm_end_bin])
            filtered_data['background'].append(smooth_data['background'][i][smooth_start_bin:smooth_end_bin])
            filtered_data['names'].append(name)

        print(f'Found {len(filtered_data["bin_content"])} matched histograms for BSM sample.')

        # Calculate Z_LR relative to smooth curves
        hypothesis_signal_function = getattr(signals, config["hypothesis_signal_function"])

        pool = Pool(processes=config["pool"])

        wksp_cfg = lambda _i: {
                'bkg_hist': filtered_data['background'][_i],
                'data': filtered_data['bin_content'][_i],
                'bin_edges': filtered_data['bin_edges'][_i],
                'W_hypo_bins': config["hypothesis_signal_width"],
                'hypo_sig_func': hypothesis_signal_function,
            }

        parameters = [(i, Workspace(wksp_cfg(i))) for i in range(len(filtered_data['background']))]

        results = list(
            tqdm(
                pool.imap(calc_z, parameters),
                total=len(parameters),
                ncols=80,
            )
        )

        pool.close() # Close Pool and let all the processes complete
        pool.join()  # Postpones the execution of next line of code until all processes in the queue are done

        to_save = {k: np.array(v, dtype=object) for k, v in filtered_data.items()}
        to_save['true_z'] = np.array(results, dtype=object)

    else:
        print(f'Preparing BSM sample located in {input_path} without smoothed backgrounds.')
        to_save = load_ATLAS(input_path, cuts=config, MC=False)

    # Save prepared data
    print(f'Writing sample to file at {output_dir}')
    for file_name, data in to_save.items(): np.save(f'{output_dir}/{file_name}.npy', data)
            
    print(f'BSM sample preparation is finished.')

    # copy config to output dir
    config_file = f'{output_dir}/config.yaml'
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--input_path', help='input root file containing BSM + SM data')
    arg_parse.add_argument('--smooth_dir', help='input directory for smoothed curves')
    arg_parse.add_argument('--output_dir', help='output directory for prepared dataset')
    
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = {**config["prepare_bsm"], **config["raw_cuts"]}

    if args.input_path is not None:
        config["input_path"] = args.input_path

    if args.smooth_dir is not None:
        config["smooth_dir"] = args.smooth_dir

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    print('>>>>> Starting BSM sample preparation step...')

    prepare_bsm(config)

    print('>>>>> Finished BSM sample preparation step.')

if __name__ == '__main__':
    main()
