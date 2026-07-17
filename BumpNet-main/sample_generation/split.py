import numpy as np
import os
import argparse
import yaml

def split(config):

    # define train/val/test fractions
    train_frac = config["train_frac"]
    val_frac = config["val_frac"]
    test_frac = config["test_frac"]
    
    if train_frac + val_frac + test_frac > 1.0:
        raise ValueError('Sum of train/val/test values must not exceed 1.')
    elif train_frac + val_frac + test_frac < 1.0:
        import warnings
        warnings.warn("Warning: Sum of train/val/test values is lower than 1.")
    
    # Create output dirs
    train_path = os.path.join(config["output_base_dir"], "train")
    val_path = os.path.join(config["output_base_dir"], "val")
    test_path = os.path.join(config["output_base_dir"], "test")
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    rng = np.random.default_rng(seed=config["seed"])
    
    for bkg_path in config["input_paths"]:
        bkg_name = bkg_path.split('/')[-1]
        print(f'Processing background: {bkg_name}')
        os.makedirs(os.path.join(train_path,bkg_name), exist_ok=True)
        os.makedirs(os.path.join(val_path,bkg_name), exist_ok=True)
        os.makedirs(os.path.join(test_path,bkg_name), exist_ok=True)
    
        # Load files
        print('loading files')
        hist_arr = np.load(f"{bkg_path}/background.npy", allow_pickle=True)
        bins_arr = np.load(f"{bkg_path}/bin_edges.npy", allow_pickle=True)
        name_arr = np.load(f"{bkg_path}/names.npy", allow_pickle=True)
    
        print(f'Loaded {len(hist_arr)} samples')
        print(f'for 1,000,000 training samples, we need {int(np.ceil(1e6/train_frac/len(hist_arr)))} copies of each sample')
        
        # Shuffle the arrays together. Idea from here: https://stackoverflow.com/a/4602224
        print('shuffling arrays')
        assert len(hist_arr) == len(bins_arr)
        assert len(hist_arr) == len(name_arr)
    
        p = rng.permutation(len(hist_arr))
    
        hist_shuffled = hist_arr[p]
        bins_shuffled = bins_arr[p]
        name_shuffled = name_arr[p]
    
        hist_train = hist_shuffled[:round(train_frac*len(hist_shuffled))]
        bins_train = bins_shuffled[:round(train_frac*len(bins_shuffled))]
        name_train = name_shuffled[:round(train_frac*len(name_shuffled))]
    
        hist_val = hist_shuffled[round(train_frac*len(hist_shuffled)):round((train_frac + val_frac)*len(hist_shuffled))]
        bins_val = bins_shuffled[round(train_frac*len(bins_shuffled)):round((train_frac + val_frac)*len(bins_shuffled))]
        name_val = name_shuffled[round(train_frac*len(name_shuffled)):round((train_frac + val_frac)*len(name_shuffled))]
    
        hist_test = hist_shuffled[round((train_frac + val_frac)*len(hist_shuffled)):round((train_frac + val_frac + test_frac)*len(hist_shuffled))]
        bins_test = bins_shuffled[round((train_frac + val_frac)*len(bins_shuffled)):round((train_frac + val_frac + test_frac)*len(bins_shuffled))]
        name_test = name_shuffled[round((train_frac + val_frac)*len(name_shuffled)):round((train_frac + val_frac + test_frac)*len(name_shuffled))]
    
        # Save the shuffled backgrounds in their correct folder
        print('saving new files\n')

        np.save(os.path.join(train_path,bkg_name,'background.npy'),hist_train,allow_pickle=True)
        np.save(os.path.join(train_path,bkg_name,'bin_edges.npy'),bins_train,allow_pickle=True)
        np.save(os.path.join(train_path,bkg_name,'names.npy'),name_train,allow_pickle=True)

        np.save(os.path.join(val_path,bkg_name,'background.npy'),hist_val,allow_pickle=True)
        np.save(os.path.join(val_path,bkg_name,'bin_edges.npy'),bins_val,allow_pickle=True)
        np.save(os.path.join(val_path,bkg_name,'names.npy'),name_val,allow_pickle=True)

        np.save(os.path.join(test_path,bkg_name,'background.npy'),hist_test,allow_pickle=True)
        np.save(os.path.join(test_path,bkg_name,'bin_edges.npy'),bins_test,allow_pickle=True)
        np.save(os.path.join(test_path,bkg_name,'names.npy'),name_test,allow_pickle=True)
    
    # copy config to output dir
    config_file = f'{config["output_base_dir"]}/config.yaml'
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)



def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--output_base_dir', help='output directory for splitted dataset')
    arg_parse.add_argument('--input_paths', nargs='+', help='lists of inputs that are split')
    
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = config["split"]
    
    if args.output_base_dir is not None:
        config["output_base_dir"] = args.output_base_dir
    
    if args.input_paths is not None:
        config["input_paths"] = args.input_paths
    
    print('>>>>> Starting splitting samples step...')

    split(config)

    print('>>>>> Finished splitting samples step.')

if __name__ == '__main__':
    main()
