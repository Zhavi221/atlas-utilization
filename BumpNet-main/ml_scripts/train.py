import os, sys
import comet_ml

try:
    from pytorch_lightning.loggers import CometLogger, CSVLogger
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint
except:
    from lightning.pytorch.loggers import CometLogger, CSVLogger
    from lightning import Trainer
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.callbacks import ModelCheckpoint

from pathlib import Path
import argparse

from lightning_ddp import DDPLightning

import glob
import yaml
import os

os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-bundle.crt' # SSL_CERT_FILE in Arch

def train(config):
    net = DDPLightning(config)
    save_dir = os.path.join(config['output_dir'], config['name'])

    debug = False # Want to debug? Set this to True

    if debug:
        print('debugging')

        trainer = Trainer(
            max_epochs = config['num_epochs'],
            accelerator='cpu',
            devices=1,
            default_root_dir=save_dir,
        )

    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        logger = None
        if 'comet' in config:
            if 'api_key' in config['comet'].keys():
                api_key = config['comet']['api_key']
            else:
                api_key = os.environ['COMET_API_KEY']
            
            if 'workspace' in config['comet'].keys():
                workspace = config['comet']['workspace']
            else:
                workspace = os.environ['COMET_WORKSPACE']

            logger = CometLogger(
                api_key=api_key,
                project=config['comet']['project_name'],
                workspace=workspace,
                offline_directory=save_dir,
                name=config['name']
            )

            net.set_comet_exp(logger.experiment)

        elif 'csv_logger' in config:
            if 'name' in config['csv_logger']:
                logger_name= config['csv_logger']['name']
            logger = CSVLogger(save_dir=save_dir, name=logger_name)

        # copy config to output dir
        config_file = f'{save_dir}/config.yaml'
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        if 'comet' in config:
            logger.experiment.log_asset(config_file,file_name='config')
            all_files = glob.glob('models/*.py') + glob.glob('models/../ml_scripts/*.py') + glob.glob('train.py')
            for fpath in all_files:
                logger.experiment.log_asset(fpath)



        # Configure callbacks (checkpoint and early stopping)
        callbacks = []
        checkpoint_callback = ModelCheckpoint(
            save_last= True,
            filename='{epoch}-{val/loss:.4f}',
            save_top_k=3,
            every_n_epochs=1,
            monitor='val/loss',
            mode='min'
        )
        callbacks += [checkpoint_callback]

        if config.get('early_stopping', False):
            early_stop = EarlyStopping(
                monitor='val/loss',
                mode='min',
                min_delta=config.get('min_delta', 1e-6),
                patience=10
            )
            callbacks += [early_stop]

        print('creating trainer')
        trainer = Trainer(
            max_epochs=config['num_epochs'],
            accelerator=config['device'],
            devices=1,
            default_root_dir=save_dir,
            logger=logger,
            callbacks = callbacks,
        )

    print('starting training')
    trainer.fit(net, ckpt_path=config['resume_from_checkpoint'])

    # Create symbolic link to best model
    best_model = trainer.checkpoint_callback.best_model_path
    best_model_link = '/'.join(Path(best_model).parts[-5:])
    best_model_link_path = f'{Path(best_model).parents[4]}/best_model.ckpt'
    # Path.symlink_to does not work for snakemake, as it does not update the date
    # so let's use bash based method:
    os.system(f'ln -sf {best_model_link} {best_model_link_path}')
    print(f'Checkpoint model saved at {best_model} (link available at {best_model_link_path})')

def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--output_dir', help='output directory for model')
    arg_parse.add_argument('--train_dir', help='input directory for trainings set')
    arg_parse.add_argument('--val_dir', help='input directory for validation set')
    arg_parse.add_argument('--seed', help='seed for training')
    
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = config["train"]
    
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    
    if args.train_dir is not None:
        config["train_dir"] = args.train_dir
    
    if args.val_dir is not None:
        config["val_dir"] = args.val_dir
    
    if args.seed is not None:
        config["seed"] = int(args.seed)
        config["name"] += "_"+str(args.seed)
    
    print('>>>>> Starting training BumpNet step...')
    
    os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-bundle.crt' # SSL_CERT_FILE in Arch
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_visible_devices']
    train(config)

    print('>>>>> Finished training BumpNet step.')

if __name__ == '__main__':
    main()
