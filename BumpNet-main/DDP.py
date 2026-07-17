import argparse
import yaml
import os
import sys

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--config', help='Config file', default='configs/default.yaml')
args = arg_parser.parse_args()


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'sample_generation'))
sys.path.append(os.path.join(project_root, 'ml_scripts'))
sys.path.append(os.path.join(project_root, 'plotting'))


with open(args.config) as f:
    config = yaml.safe_load(f)

tasks = config['tasks']

if tasks.get('do_smooth', False):
    print('>>>>> Starting smoothing step...')
    from sample_generation import smooth
    config_smooth = {**config["smooth"], **config["raw_cuts"]}
    smooth.smooth(config_smooth)
    print('>>>>> Finished smoothing step.')

if tasks.get('do_function', False):
    print('>>>>> Starting generating function backgrounds step...')
    from sample_generation import function
    function.function(config["function"])
    print('>>>>> Finished generating function backgrounds step.')

if tasks.get('do_split', False):
    print('>>>>> Starting splitting backgrounds step...')
    from sample_generation import split
    split.split(config["split"])
    print('>>>>> Finished splitting backgrounds step.')

if tasks.get('do_stretch', False):
    print('>>>>> Starting stretching background step...')
    from sample_generation import stretch
    stretch.stretch(config["stretch"])
    print('>>>>> Finished stretching background step.')

if tasks.get('do_systematics', False):
    print('>>>>> Starting systematics step...')
    from sample_generation import systematics
    systematics.systematics(config["systematics"])
    print('>>>>> Finished systematics step.')

if tasks.get('do_generate', False):
    print('>>>>> Starting generating samples step...')
    from sample_generation import generate
    generate.generate(config["generate"])
    print('>>>>> Finished generating samples step.')

if tasks.get('do_prepare_bsm', False):
    print('>>>>> Starting BSM sample preparation step...')
    from sample_generation import prepare_bsm
    config_prepare_bsm = {**config["prepare_bsm"], **config["raw_cuts"]}
    prepare_bsm.prepare_bsm(config_prepare_bsm)
    print('>>>>> Finished BSM sample preparation step.')

if tasks.get('do_merge', False):
    print('>>>>> Starting merging step...')
    from sample_generation import merge
    merge.merge(config["merge"])
    print('>>>>> Finished merging step.')

if tasks.get('do_train', False):
    print('>>>>> Starting training step...')
    # Set TLS CA certificate bundle
    os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-bundle.crt' # SSL_CERT_FILE in Arch
    os.environ["CUDA_VISIBLE_DEVICES"] = config["train"]['cuda_visible_devices']
    from ml_scripts import train
    train.train(config['train'])
    print('>>>>> Finished training step.')

if tasks.get('do_prepare', False):
    print('>>>>> Starting input preparation step...')
    from sample_generation import prepare
    for dataset_config in config['prepare']:
         prepare.prepare(dataset_config)
    print('>>>>> Finished input preparation step.')

if tasks.get('do_predict', False):
    print('>>>>> Starting prediction step...')
    from ml_scripts import predict
    config_pred = {**config["predict"], **config["raw_cuts"]}
    predict.predict(config_pred)
    print('>>>>> Finished prediction step.')

if tasks.get('do_merge_BN_output', False):
    print('>>>>> Starting merging step...')
    from ml_scripts import mergeBNoutput
    mergeBNoutput.mergeBNoutput(config['mergeBNoutput'])
    print('>>>>> Finished merging step.')

if tasks.get('do_plot', False):
    print('>>>>> Starting plotting step...')
    from plotting import plot
    config_plot = {**config["plot"], **config["raw_cuts"]}
    plot.plot(config_plot)
    print('>>>>> Finished plotting step.')

if tasks.get('do_generate_toys_bkg_only', False):
    print('>>>>> Starting background-only toy generation step...')
    from global_significance import generate_toys_bkg_only          # new module
    generate_toys_bkg_only.generate_toys_bkg_only(config['generate_toys_bkg_only'])
    print('>>>>> Finished background-only toys generation step.')

if tasks.get('do_predict_toys_bkg_only', False):
    print('>>>>> Starting BumpNet prediction on toy samples...')
    from global_significance import predict_toys_bkg_only
    predict_toys_bkg_only.predict_toys_bkg_only(config['predict_toys_bkg_only'])
    print('>>>>> Finished BumpNet prediction on toys.')

if tasks.get('do_plot_toys_bkg_only', False):
    print('>>>>> Starting toy-only plotting step...')
    from global_significance import plot_toys_bkg_only
    plot_toys_bkg_only.plot_toys_bkg_only(config['plot_toys_bkg_only'])
    print('>>>>> Finished toy-only plotting step.')

if tasks.get('do_analyze_split_ds', False):
    print('>>>>> Starting split-DS analysis step...')
    from global_significance import analyze_split_ds
    analyze_split_ds.analyze_split_ds(config['analyze_split_ds'])
    print('>>>>> Finished split-DS analysis step.')