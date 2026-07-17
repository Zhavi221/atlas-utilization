import os
import sys
import glob
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import argparse
import yaml

import torch
from torch.utils.data import DataLoader

from dataset import DDPDataset, DdpSampler, collate_fn_pred
from models.Znet3 import Znet


def predict(config):

    # load the trained network
    net = Znet()
    checkpoint_path = config['checkpoint_path']


    print('\ncheckpoint loaded from -', checkpoint_path, '\n')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k.replace('net.', '')] = v
    net.load_state_dict(state_dict)

    net.eval()

    # loop over all files to evaluate on
    for input_dir, output_dir in zip(config["input_dirs"], config["output_dirs"]):
        print(f'running prediction on -\n{input_dir}')


        # create the dir (if not already exist)
        pred_dir = os.path.join(output_dir)
        Path(pred_dir).mkdir(parents=True, exist_ok=True)
        
        # copy config to output dir
        config_file = f'{output_dir}/config.yaml'
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


        print('\npredictions saved at -', pred_dir, '\n')

        # dataloader

        ds = DDPDataset(config, input_dir, mode='predict')
        batch_sampler = DdpSampler(config, ds.n_bins, batch_size=config['batch_size'], shuffle=False)
        loader = DataLoader(ds, num_workers=config['num_workers'],
            collate_fn=collate_fn_pred, batch_sampler=batch_sampler, pin_memory=False)


        n_hist = len(ds.bin_content)
        z_pred = np.empty(n_hist, dtype=object)
        b_pred = np.empty(n_hist, dtype=object)

        for idx, x in tqdm(loader):
            z, b = net(x)
            z_pred[idx] = [z_i.detach().numpy() for z_i in z]
            b_pred[idx] = [b_i.detach().numpy() for b_i in b]

        now = datetime.now()
        timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
        extra_info = {
            'timestamp': timestamp,
            'checkpoint_path': checkpoint_path
        }

        zf_name = os.path.join(pred_dir, f'pred_z')
        np.save(zf_name + '.npy', z_pred)

        bf_name = os.path.join(pred_dir, f'pred_b')
        np.save(bf_name + '.npy', b_pred)

        print('done\n')

def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--ckpt', help='Checkpoint path')
    arg_parse.add_argument('--output_dirs', nargs='+', help='output directory for prediction')
    arg_parse.add_argument('--input_dirs', nargs='+', help='list of input directory for testing')
    arg_parse.add_argument('--checkpoint_path', help='path to the model')
    
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = {**config["predict"], **config["raw_cuts"]}
    
    if args.output_dirs is not None:
        config["output_dirs"] = args.output_dirs
    
    if args.input_dirs is not None:
        config["input_dirs"] = args.input_dirs
        
    if args.checkpoint_path is not None:
        config["checkpoint_path"] = args.checkpoint_path

    print('>>>>> Starting evaluating BumpNet step...')

    predict(config)

    print('>>>>> Finished evaluating BumpNet step.')

if __name__ == '__main__':
    main()
