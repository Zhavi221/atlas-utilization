import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from copy import deepcopy
import subprocess
import glob
import yaml
import argparse

def merge(config):

    # Create output directory
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initiate random generator
    rng = np.random.default_rng(config["seed"])

    # Loop over different file types and combine them
    for x in ['background', 'names', 'wanted_z_and_mu', 'bin_edges', 'signal_shape', 'bin_content', 'true_z']:
        input_files = [f'{str(Path(input_dir))}/{x}.npy' for input_dir in config["input_paths"]]
        # Expand the '*' in file, if applicable
        files = []
        for f in input_files:
            files += glob.glob(f)

        # Load and stack .npy arrays
        arrays = []
        for f in files:
            print(f'Adding {f} file...')
            arrays.append(np.load(f, allow_pickle=True))

        if x == 'names':
            for i,arr in enumerate(arrays):
                arrays[i] = arr.flatten()


        merged = np.concatenate(arrays, axis=0)

        # Shuffle lines in combined file
        if config["shuffle"]:
            print('Shuffling...')
            indices = rng.permutation(len(merged))
            merged = merged[indices]

        # Save merged array
        np.save(f'{output_dir}/{x}.npy', merged) 

    print(f'Sample merging finished')
    # copy config to output dir
    config_file = f'{output_dir}/config.yaml'
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)



def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--output_dir', help='output directory for merged dataset')
    arg_parse.add_argument('--input_paths', nargs='+', help='lists of inputs that are merged')
    
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = config["merge"]
    
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    
    if args.input_paths is not None:
        config["input_paths"] = args.input_paths
    
    print('>>>>> Starting splitting samples step...')

    merge(config)

    print('>>>>> Finished splitting samples step.')

if __name__ == '__main__':
    main()





