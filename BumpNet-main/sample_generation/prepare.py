import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
from timeit import default_timer
from os import makedirs
from tqdm import tqdm

import ROOT

def prepare(config):

    verbose=False

    print(f'> Preparing {config["input_file"]}...')

    # Prepare output
    outdir = config['output_dir'] if 'output_dir' in config else 'input_data'
    makedirs(outdir, exist_ok=True)

    # Start
    tic = default_timer()

    # Open ROOT file
    root_file = config['input_file']
    tfile = ROOT.TFile.Open(root_file ,"READ")

    # Process histograms
    hists_data = {'bin_content' : [], 'bin_edges' : [], 'names' : []}
    hists_names = [k.GetName() for k in tfile.GetListOfKeys()]
    print(f'> Processing {len(hists_names)} histograms...')
    if 'cuts' in config and config['cuts'].get('no_massT', False):
        print('> Removing massT histograms')
    for hist_name in tqdm(hists_names):

        if 'hCat' not in hist_name: continue
        if 'cuts' in config:
            if config['cuts'].get('no_massT', False) and 'massT' in hist_name:
                continue

        # Get binning with mass values
        th1 = tfile.Get(hist_name)
        if not th1: print(f"Failed to get histogram\n hist_name = {hist_name}\n root_file = {root_file}")
        th1.SetDirectory(0)
        bin_content = np.array([th1.GetBinContent(b) for b in range(1,th1.GetXaxis().GetNbins()+1)])
        bin_edges = np.array(list(th1.GetXaxis().GetXbins()))
        bin_sizes = np.array([bin_edges[i+1]-bin_edges[i] for i in range(0, len(bin_edges)-1)])
        bin_centers = bin_edges[:-1] + bin_sizes/2


        hists_data['bin_content'] += [bin_content]
        hists_data['bin_edges'] += [bin_edges]
        hists_data['names'] += [np.array([hist_name])]

    # Close ROOT file
    tfile.Close()


    # Print statistics
    nbins = np.array([len(h) for h in hists_data['bin_content']])
    ymin = np.min(np.array([np.min(h) for h in hists_data['bin_content']]))
    ymax = np.max(np.array([np.max(h) for h in hists_data['bin_content']]))
    xmin = np.min(np.array([np.min(h) for h in hists_data['bin_edges']]))
    xmax = np.max(np.array([np.max(h) for h in hists_data['bin_edges']]))
    print(f'> Total of {len(hists_data["X"])} histograms with\n'
          f'  {np.min(nbins)} <= Nbins <= {np.max(nbins)}\n'
          f'  {ymin} <= entries per bin <= {ymax}\n'
          f'  {xmin} <= mass <= {xmax}'
          )

    # Save histograms into DDP input format
    # TODO: add header
    for tag, data in hists_data.items():
        out_file = f'{outdir}/{tag}.npy'
        np.save(out_file, np.array(data, dtype=object))
        print(f'> Saved histograms to {out_file}')

    # Finished
    toc = default_timer()
    print(f'Elapsed time: {toc - tic}s')
