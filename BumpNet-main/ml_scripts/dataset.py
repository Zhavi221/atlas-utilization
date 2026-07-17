from pathlib import Path
import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utilities.data_loading import load_ATLAS


class DDPDataset(Dataset):

    def __init__(self, config, dir_path, mode='train', compute_scale=False):

        '''
            mode: train, val, predict
        '''

        self.config = config
        self.mode   = mode
        self.scale  = None

        # check which format we are using (npy recommended for variable number of bins)
        print(f'Loading data from {dir_path} ...')

        paths = list(Path(dir_path).rglob(f'*.*'))
        # Quick loop to remove unwanted files
        for path in paths:
            input_format = path.suffix.lstrip('.')
            if input_format.endswith("npy")\
            or input_format.endswith("root"):
                break
        if input_format is None:
            print(f'input_format not found !')

        if self.mode == 'predict':
            self.cuts = {'min_num_events': config['min_num_events'], 
                    'min_num_bins': config['min_num_bins'],
                    'skipped_bins': config['skipped_bins'],
                    'min_total_events': config['min_total_events']}
            self.bin_content = self.read_data(dir_path, input_format=input_format, which_data='bin_content')

        elif (self.mode == 'train') or (self.mode == 'val'):
            self.bin_content = self.read_data(dir_path, input_format=input_format, which_data='bin_content')
            self.true_z = self.read_data(dir_path, input_format=input_format, which_data='true_z')
            self.background = self.read_data(dir_path, input_format=input_format, which_data='background')

            self.scale = self.compute_scale()

        # needed for sampler
        self.n_bins = [np.array(x).shape[0] for x in self.bin_content]
        self.n_bins = np.array(self.n_bins)

        self.n_example = self.bin_content.shape[0]

        print('number of histograms', self.n_example)


    def read_data(self, dir_path, input_format, which_data='bin_content', num_rows=None):

        if input_format == 'npy':
            path = list(Path(dir_path).glob(f'**/{which_data}.npy'))[0]
            return np.load(path, allow_pickle=True)

        elif input_format == 'root':
            # Save data dictionary as attribute to avoid repeated loading
            if self.data is None:
                self.data = load_ATLAS(Path(f'{dir_path}/rebinned.root'), cuts=self.cuts)
            return self.data[which_data]
        
        else:
            raise ValueError(f"The input must be a .npy or .root file, instead got: {input_format}")

    def compute_scale(self):
        scale = {}
        for type_f in ['bin_content', 'true_z', 'background']:
            data = np.hstack(getattr(self, type_f.lower()))
            scale[type_f.lower()] = {
                'min': data.min(),
                'max': data.max()
            }
        return scale


    def __len__(self):
        return self.n_example


    def __getitem__(self, idx):

        obs_np = np.array(self.bin_content[idx], dtype=np.float32)
        obs = torch.from_numpy(obs_np)


        if self.mode == 'predict':
            return idx, obs

        z_np = np.array(self.true_z[idx], dtype=np.float32)
        z = torch.from_numpy(z_np)

        b_np = np.array(self.background[idx], dtype=np.float32)
        b = torch.from_numpy(b_np)

        return obs, z, b



class DdpSampler(Sampler):

    def __init__(self, config, n_bins_array, batch_size, shuffle=True):
        '''
        Args
            n_bins_array: array of number of bins in the histograms
            batch_size  : batch size
        '''

        super().__init__(n_bins_array.size)

        self.config = config
        self.shuffle = shuffle
        self.seed = config['seed']
        self.rng = np.random.default_rng(self.seed)
        
        self.dataset_size = n_bins_array.size
        self.batch_size = batch_size

        self.drop_last = False

        self.index_to_batch = {}
        self.bin_size_idx = {}
        running_idx = -1

        for n_bins_i in set(n_bins_array):

            self.bin_size_idx[n_bins_i] = np.where(n_bins_array == n_bins_i)[0]

            n_of_size = len(self.bin_size_idx[n_bins_i])
            n_batches = max(n_of_size / self.batch_size, 1)
            
            self.bin_size_idx[n_bins_i] = np.array_split(self.rng.permutation(self.bin_size_idx[n_bins_i]),
                                                           n_batches)
            for batch in self.bin_size_idx[n_bins_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.arange(self.n_batches)
        if self.shuffle:
            batch_order = self.rng.permutation(batch_order)
        for i in batch_order:
            yield self.index_to_batch[i]



def collate_fn(samples):
    xs = [x[0].unsqueeze(0) for x in samples]
    zs = [x[1].unsqueeze(0) for x in samples]
    bs = [x[2].unsqueeze(0) for x in samples]

    xs = torch.cat(xs)
    zs = torch.cat(zs)
    bs = torch.cat(bs)

    return xs, zs, bs



def collate_fn_pred(samples):
    idxs = [x[0] for x in samples]
    xs   = [x[1].unsqueeze(0) for x in samples]

    xs   = torch.cat(xs)

    return idxs, xs

