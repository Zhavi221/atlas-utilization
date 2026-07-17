import os
import json
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import copy


# get cwd
cwd = os.path.dirname(os.path.abspath(__file__))



arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', '--config', help='Config file')
args = arg_parser.parse_args()


with open(args.config) as f:
    config = json.load(f)


resource = 'CPU'

# make sure we don't mix CPU and GPU tasks
tasks = config['tasks']
if tasks['do_train'] or tasks['do_predict']:
    resource = 'GPU'
    if tasks['do_smooth'] or tasks['do_function'] or tasks['do_generate']:
        raise ValueError('train and predict use GPU. smooth, function, generate use CPU. Cannot mix them.')
    
if tasks['do_smooth'] or tasks['do_function'] or tasks['do_generate']:
    if tasks['do_train'] or tasks['do_predict']:
        raise ValueError('train and predict use GPU. smooth, function, generate use CPU. Cannot mix them.')


if resource == 'CPU':
    print("\033[36mAssuming that we are running (do_smooth or do_function or both) and do_generate\033[0m")
    print("\033[36mExtremely over-engineered; may break at any moment :(\033[0m")

    ncpus = '1'
    ngpus = '0'
    mem   = '1gb'
    walltime = '11:00:00'

    seeds = np.arange(101, 101 + config['wis_stuff']['njobs_for_generation'], 1)
    print('submitting jobs...')
    for seed in tqdm(seeds):
        dir_seed = os.path.join(config['wis_stuff']['base_dir_for_generation'], f'seed_{seed}')
        os.makedirs(dir_seed, exist_ok=True)

        # make a detached copy of the config
        config_seed = copy.deepcopy(config)
        config_seed['function']['seed'] = int(seed)
        config_seed['function']['output_dir'] = os.path.join(dir_seed, config['function']['output_dir'])

        config_seed['generate']['seed'] = int(seed)
        config_seed['generate']['output_dir'] = os.path.join(dir_seed, config['generate']['output_dir'])
        for k,v in config_seed['generate']['backgrounds'].items():
            config_seed['generate']['backgrounds'][k]['hist'] = os.path.join(dir_seed, v['hist'])
            config_seed['generate']['backgrounds'][k]['name'] = os.path.join(dir_seed, v['name'])
            config_seed['generate']['backgrounds'][k]['binning'] = os.path.join(dir_seed, v['binning'])

        config_seed['smooth']['output_dir'] = os.path.join(dir_seed, config['smooth']['output_dir'])
        config_seed['stretch']['output_dir'] = os.path.join(dir_seed, config['stretch']['output_dir'])

        for dataset, dataset_seed in zip(config['plot']['datasets'], config_seed['plot']['datasets']):
            dataset_seed['output_dir'] = os.path.join(dir_seed, dataset['output_dir'])
            dataset_seed['sig_input_dir'] = os.path.join(dir_seed, dataset['sig_input_dir'])
            dataset_seed['bkg_input_dir'] = os.path.join(dir_seed, dataset['bkg_input_dir'])
            if dataset_seed['sig_prediction_dir']: dataset_seed['sig_prediction_dir'] = os.path.join(dir_seed, dataset['sig_prediction_dir'])
            if dataset_seed['bkg_prediction_dir']: dataset_seed['bkg_prediction_dir'] = os.path.join(dir_seed, dataset['bkg_prediction_dir'])

        new_config_path = os.path.join(dir_seed, 'config.json')
        with open(new_config_path, 'w') as f:
            json.dump(config_seed, f, indent=4)

        command  = f'qsub -o {dir_seed}/output.log'
        command += f' -e {dir_seed}/error.log'
        command += f' -q N -N ddp -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io=1'
        command += f' -v CONFIG={new_config_path},CWD={cwd}'
        command += f' {os.path.join(cwd, "run_on_node_pbs.sh")}'

        # print(command)
        os.system(command)

    print('done')

elif resource == 'GPU':

    if tasks['do_train']:
        shutil.copy(args.config, config['train']['output_dir'])

    ncpus = '4'
    ngpus = '1'
    mem   = '20gb'
    walltime = '72:00:00'

    command  = f'qsub -o {config["train"]["output_dir"]}/output.log'
    command += f' -e {config["train"]["output_dir"]}/error.log' 
    command += f' -q gpu -N ddp -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus}'
    command += f' -v CONFIG={args.config},CWD={cwd}'
    command += f' {os.path.join(cwd, "run_on_node_pbs.sh")}'


    print(command)
    os.system(command)

