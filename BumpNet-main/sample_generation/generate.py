import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import signals
from workspace import Workspace
import pandas as pd
from copy import deepcopy
import argparse
import yaml



def inject_sig_and_scan(parameters):

    i, wksp, signal_mean, wanted_z = parameters

    # Initialize injected signal histogram
    wksp.M_inj        = wksp.bin_centers[signal_mean]
    wksp.W_inj        = wksp.bin_widths[signal_mean]*wksp.W_inj_bins
    wksp.inj_sig_hist = wksp.asimov(wksp.inj_sig_func, params=(wksp.M_inj,wksp.W_inj))

    # Initialize signal hypothesis histogram
    wksp.M_hypo        = wksp.bin_centers[signal_mean]
    wksp.W_hypo        = wksp.bin_widths[signal_mean]*wksp.W_hypo_bins
    wksp.hypo_sig_hist = wksp.asimov(wksp.hypo_sig_func, params=(wksp.M_hypo,wksp.W_hypo))

    # Add signal with wanted strength to the unfluctuated background
    wksp.mu = wksp.calc_mu_for_wanted_z(wanted_z) if wanted_z > 0 else 0.0

    if wksp.mu is None:
        return 'skip'

    # Save these for later (return)
    mu_inj = wksp.mu
    sig_inj = deepcopy(wksp.inj_sig_hist)

    # Create the data by fluctuating bkg + mu_inj*sig (checking that it doesn't go below 1)
    wksp.data = wksp.sample()

    # Scan over M in bin centers and test significance of signal centered there
    z_scan = wksp.z_scan()

    if type(z_scan) == str:
        if z_scan == 'skip':
            return 'skip'

    return i, wksp.data, mu_inj, z_scan, sig_inj


def generate(config):

    print(f'Start sample generation for {config["output_dir"]}...')
    # Create output directory
    output_dir = f'{config["output_dir"]}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initiate random generator
    rng = np.random.default_rng(config["seed"])

    # Load all backgrounds
    bkg = []
    bin_edges = []
    names = []
    for bkg_path in config["input_paths"]:


        # Load background histograms from file
        bkg_i = np.load(f"{bkg_path}/background.npy", allow_pickle=True)
        bin_edges_i = np.load(f"{bkg_path}/bin_edges.npy", allow_pickle=True)
        name_i = np.load(f"{bkg_path}/names.npy", allow_pickle=True)

        if bkg_i is None:
            continue

        nbins_max_i = max(len(hist) for hist in bkg_i)
        print(f'Loaded {bkg_path} backgrounds with {len(bkg_i)} histograms and nbins_max = {nbins_max_i}')

        bkg.append(bkg_i)
        bin_edges.append(bin_edges_i)
        names.append(name_i)


    bkg = np.concatenate(bkg, axis=0)
    bin_edges = np.concatenate(bin_edges, axis=0)
    names = np.concatenate(names, axis=0)


    nhists_loaded = bkg.shape[0]
    print(f'Loaded total of {nhists_loaded} background histograms')
    print(f'Want {config["number_of_samples"]} histograms')

    fluctuations = int(np.round(config["number_of_samples"]/nhists_loaded))
    if fluctuations == 0:
        fluctuations = 1


    remove_NaN = lambda row : row[np.isfinite(row)]
    bkg = np.array([remove_NaN(np.array(row, dtype=float)) for row in bkg], dtype=object)
    bin_edges = np.array([remove_NaN(np.array(row, dtype=float)) for row in bin_edges], dtype=object)

    postfix = np.vstack([np.arange(1,fluctuations+1,1).astype(str)[:,None] for i in range(bkg.shape[0])])
    bkg = np.repeat(bkg, repeats=fluctuations, axis=0)
    bin_edges = np.repeat(bin_edges, repeats=fluctuations, axis=0)
    names = np.repeat(names, repeats=fluctuations, axis=0)
    postfix = postfix.reshape(names.shape)
    names = names + '_' + postfix

    nhists  = bkg.shape[0]
    print(f'Repeated each histogram {fluctuations} times for a total of {nhists} background histograms')

    # Randomly sample significances
    wanted_z = [rng.uniform(config["z_range"][0],config["z_range"][1]) for i in range(bkg.shape[0])]
    wanted_z = np.asarray(wanted_z)

    # Set percentage of examples that will have signal
    add_signal = rng.choice([1, 0], size=bkg.shape[0], p=[config["percent_with_signal"], 1-config["percent_with_signal"]])

    # Obtain injected signal function
    inj_sig_type = config["injected_signal_function"]
    inj_signal_function = getattr(signals, inj_sig_type)

    # Obtain signal hypothesis function
    hypo_sig_type = config["hypothesis_signal_function"]
    hypo_signal_function = getattr(signals, hypo_sig_type)

    print(f'Generating sample with {inj_sig_type} injected signal, using a {hypo_sig_type} signal hypothesis.')

    # Sample random numbers for the position of each signal shape
    nbins = [b.shape[0] for b in bkg]

    signal_mean = [rng.choice(np.arange(int(config["edge"][0]*n),int(config["edge"][1]*n)), size=1) for n in nbins]

    # Generate seeds for the poisson fluctuation of each subprocess
    # This is to make sure they are not equally seeded.
    pois_ss = np.random.SeedSequence(config["seed"])
    poisson_seeds = pois_ss.spawn(bkg.shape[0])


    # Set background and signal combination in parallel
    pool = Pool(processes=config["pool"])
    
    # Returns the configuration for each workspace
    wksp_cfg = lambda _i: {
        'seed': poisson_seeds[_i],
        'bkg_hist': bkg[_i],
        'bin_edges': bin_edges[_i],
        'W_inj_bins': config["injected_signal_width"], # injected signal width in bins
        'W_hypo_bins': config["hypothesis_signal_width"], # signal hypothesis width in bins; default is 1.0
        'inj_sig_func': inj_signal_function,
        'hypo_sig_func': hypo_signal_function,
    }
    
    parameters = [(i,
                    Workspace(wksp_cfg(i)),     #workspace
                    signal_mean[i],             #position of injected sig
                    wanted_z[i]*add_signal[i],  #significance of injected sig
                    ) for i in tqdm(range(bkg.shape[0]))]
    
    results = list(
        tqdm(
            pool.imap(inject_sig_and_scan, parameters),
            total=len(parameters),
            ncols=80,
        )
    )
    
    pool.close() # Close Pool and let all the processes complete
    pool.join()  # Postpones the execution of next line of code until all processes in the queue are done

    # pop out skipped results
    results = np.array(results, dtype=object)
    if len(results.shape) == 1:
        results_indexing = (results != 'skip')
        number_of_skips = (results == 'skip').sum()
        print(f'skipped {number_of_skips} histograms')

        results = results[results_indexing]
        wanted_z = wanted_z[results_indexing]
        names = names[results_indexing]
        bkg = bkg[results_indexing]
        bin_edges = bin_edges[results_indexing]


    # Combine results for obs, mu3, wanted_mu, z
    # Note: Pool returns in the same order as the parameters were provided
    idx = np.array([r[0] for r in results])
    bin_content = np.array([r[1] for r in results], dtype=object)
    wanted_mu = np.array([r[2] for r in results])
    true_z = np.array([r[3] for r in results], dtype=object)
    signal_shape = np.array([r[4] for r in results], dtype=object)
    wanted_z_and_mu = np.array([np.array([i,j]) for i,j in zip(wanted_z, wanted_mu)], dtype=object)

    print(f'Writing sample to file at {output_dir}')
    for file_name, data in {'wanted_z_and_mu' : wanted_z_and_mu,
                            'bin_content' : bin_content,
                            'true_z' : true_z,
                            'signal_shape' : signal_shape,
                            'background' : bkg,
                            'bin_edges' : bin_edges,
                            'names' : names}.items():
        np.save(f'{output_dir}/{file_name}.npy', data)
            
    print(f'Sample generation finished')

    # copy config to output dir
    config_file = f'{output_dir}/config.yaml'
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--seed', help=' random seed')
    arg_parse.add_argument('--percent_with_signal', help='signal percentage')
    arg_parse.add_argument('--output_dir', help='output directory for generated dataset')
    arg_parse.add_argument('--input_paths', nargs='+', help='lists of inputs that are first merged and then used for generation')
    arg_parse.add_argument('--number_of_samples', help='number of wanted samples')
    
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = config["generate"]

    if args.seed is not None:
        config["seed"] = int(args.seed)
    
    if args.percent_with_signal is not None:
        config["percent_with_signal"] = float(args.percent_with_signal)
    
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    
    if args.input_paths is not None:
        config["input_paths"] = args.input_paths
    
    if args.number_of_samples is not None:
        config["number_of_samples"] = int(args.number_of_samples)

    print('>>>>> Starting generating samples step...')

    generate(config)

    print('>>>>> Finished generating samples step.')

if __name__ == '__main__':
    main()
