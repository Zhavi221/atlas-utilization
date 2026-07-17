# Data Directed Paradigm framework

## Structure description

* `configs`: Configuration files, formatted as yaml files
* `sample_generation` : Python scripts that generate the training, validation and application data
* `ml_scripts`: Model architecture definitions
* `plotting` : Python scripts that plot various figures
* `notebooks`: jupyter notebooks 
* `tutorials`: instructions on how to reproduce certain results


## Workflow

This framework can be used to perform the following tasks:

* Smooth pre-existing histogram into analytical functions
* Generate synthetic data by injecting signal into background histograms
* Train a DDP model
* Evaluate a DDP model 

For all of the tasks, the workflow has 2 steps: set the configuration file
and submit the job.

### Step 1: Configuration

Configuration files can be found inside the `configs` folder. An example is available in the folder. You can copy any of them and modify it according to your needs. See the [Configuration description](#config) section for a more in-depth view at each task and the various optionsfor each of them.


### Step 2: Submission
`
Once a configuration file is ready, it can be submitted as a command-line argument while calling the `DDP.py` file. The following command could then be used to run the code:

`python DDP.py --config <your-config-file>`

A problem with this, however, is that the code can use _a lot_ of computational resources and we recommend running the code on clusters. As an example, the files `submit_rorqual.sh`, `submit_rorqual_multinodes.sh` and `run_on_node_pbs.sh` are all submission scripts design to submit a BumpNet config file to a local cluster (respectively Calcul Québec's Rorqual cluster and Weizmann Institute of Science's PBS cluster). We strongly encourage anyone using this code to write their own submission scripts

These files are a bash scripts containing different commands to run all the framework functionalities, namely: generate, merge, train, evaluate and plot. They also sets memory and time allocations, needed when sending within a cluster. Modify it according to your needs.

Running the DDP framework requires specific python Packages. For setting up the right dependencies, a conda virtual environment can be created. For this, first create and activate the environment 
```
conda create --name DDP python=3.11
conda activate DDP
```
Then the required dependency can be installed via 
```
pip install -r requirements.txt
```


## Configuration description: {#config}

All the different steps of a BumpNet analysis (smooth, functions, generate, train, predict, plot) are controlled via a unique yaml config file. The default one is `configs/baseline_DMfunc.yaml`. You can choose to do only part of the workflow by turning to`true` the tasks that you want to do in the beginning of the config file.

```yaml
tasks:
  do_smooth: false
  do_function: false
  do_split: false
  do_stretch: false
  do_systematics: false
  do_generate: false
  do_train: false
  do_predict: false
  do_plot: false
```


Now, let's explore the various tasks that the DDP can do, with the help of the config file.

### Cuts

There are some cuts that we want to apply to the raw histograms so that we can ensure behaviour as expected with BumpNet. The cut values are defined globally in the config so that the same ones are applied in every step.

```yaml
cut:
  min_num_bins: 25
  min_total_events: 100
  min_num_events: 0.35
  skipped_bins: 0.1
```

### Smoothing

The smoothing step smooth MC data for future use. This step is required so that we can generate more examples of realistic data in a later step. The smoothing is controlled in this block of code 

```yaml
smooth:
  input_path: input-dir/rebinned.root
  output_dir: temp/data
  nominal_dir: temp/data_nom
  seed: 42
  smooth_method: ''
  verbose: false
```

- `output_dir` is the directory you want the smoothed sample to go to. It does not have to already exist, and will be in the data_directed_samples mother directory.

- `smooth-method` allows the use of different techniques to smooth the data. The options are "func-forms", "gpr", or "" which uses a hybrid method. The default should be "".

- `input_path` is the path to the histograms that you want to smooth.

- The `fit` section is in case you want to impose cuts on the quality of the smoothing. We currently don't use any.

- (optional) `nominal_dir` is used for systematic variations when the smoothing of a nominal sample is available. If so, the smoothing algorithm only smooths histograms that are in both `input-dir/rebinned.root` and `temp/data` and additionally ensures consistent bin edges. 

This step will create thee files, a your_name_func.npy (the smoothed histograms), a your_name_name.npy (the names/categories of the smoothed histograms), and a your_name_bins.npy (the bin numbers of the smoothed histograms).


### Functions

The functions are controlled in this block of code

```yaml
function:
  smooth_dir: baseline/smooth
  output_dir: baseline/function_bkg
  background_functions:
  - exponential
  - linear
  - one_over_x
  - one_over_x_squared
  - one_over_x_cubed
  - one_over_x_to_4th
  - one_over_x_to_nth
  - parabola_half
  - ln_negative
  - cos_quarter
  - cosh_half
  seed: 42
  custom_parameters:
    nbins:
    - 30
    - 100
    min_x: 1
    max_x: 4000
    min_ymin: 0.1
    max_ymin: 1
    min_ymax: 3
    max_ymax: 1725000
    number_of_samples: 1000
  rescale_minimum: 0
```

- `smooth_dir` is the directory where the smoothed curves are located, from which the dynamic range of the functions is calculated. If this is not specified, default values will be used.

- `output_dir` is the directory you want the functions sample to go to. It does not have to already exist, and will be in the data_directed_samples mother directory.

- `background_functions` are the functions you want to include in your sample. The current possibilities are already there.

- `seed` is the random seed.

- `number_of_samples` : the number of samples you want from each function. 

- `rescale_minimum` : rescales the minimum of all functions to a specified value. Setting this to `0` turns the feature off. 

- `custom_parameters` (optional) : this can be used to specify a custom dynamic range for the functions, if desired. 

This step will create thee files, a bakcground.npy (the functions histograms), a names.npy (the names/categories of the functions histograms), and a bin_edges.npy (the bin numbers of the functions histograms).

### Split

The split step is used to split the dataset into training, testing and validation in order to ensure independent robustness tests.

```yaml
split:
  output_base_dir: baseline/bkg_split
  train_frac: 0.8
  val_frac: 0.1
  test_frac: 0.1
  seed: 42
  input_paths: ["baseline/DM_bkg_smoothed", "baseline/function_bkg"]
```
### Generate

The generate step is used to merge different datasets and then generate realistic examples out of them. It first generates a signal, inject it in the smooth bkg, fluctuates bkg + signal and computes the statistical significance. The generation is controlled via this block of code 

```yaml
generate:
  output_dir: baseline/bkg_split/train/DM_and_func_bkg_with_gaussian_signal_seed_1_percent_with_signal_1.0
  input_paths: ["baseline/bkg_split/train/DM_bkg_smoothed", "baseline/bkg_split/train/function_bkg"]
  injected_signal_function: gaussian
  hypothesis_signal_function: gaussian
  seed: 1
  pool: 40
  injected_signal_width: 1.0
  hypothesis_signal_width: 1.0
  edge:
  - 10
  - 0
  z_range:
  - 1
  - 10
  number_of_samples: 1000000.0
  optimization_tolerance: 0.0001
```

- `output_dir` is the directory you want the functions sample to go to. It does not have to already exist, and will be in the data_directed_samples mother directory.

- `output_format` is either "npy" or "npy". Please be coherent in all the sections of the config file.

- `backgrounds` refer to the background shapes you want to use. They can either be functions (`function`) or smoothed dark machines histograms (`dark_machines`). You input the files previously created in the `smoothing` and `functions` steps. Note that this can contain as many datasets as you want, their names does not matter.

- `injected_signal_function` is the shape of the artificial signal injected. We currently have only gaussian.

- `hypothesis_signal_function` is the shape of the signal used for the signal + background hypothesis in the $Z_{LR}$ calculation. By default, it is gaussian. It should remain untouched.

- `seed` is the random seed used for generating the sample. You can put multiple seeds in [] to get multiple samples with different seeds (ie putting three different seeds will give you three different datasets of the same size).

- `pool` is the number of processes happening in parallel. It is dependant on the cluster.

- `injected_signal_width` is the number of bins of the histogram that correspond to one sigma of the injected signal. Standard is 1 but tests have been done up to 3.

- `hypothesis_signal_width` is the number of bins of the histogram that correspond to one sigma of the signal used in the signal + background hypothesis. Default is 1. It should remain untouched.

- `percent_with_signal` is the fraction of the generated histograms that you want to have an artificial gaussian signal injected in. Setting `"percent_with_signal" : [1.0, 0.0]` will create two samples : one with signal in 100% of the histograms, and one backgrounds only (0% injected signal).

- `edge` is the number of bins on each side of the histogram in which you don't want to inject the signal. Setting `"edge":[6,8]` ensures that the gaussian signal in not injected in the first 6 bins or last 8 bins of the histograms. This is usually set to [0, 0] such that signals are injected everywhere.

- `z range` is the range of the significances of the artificial gaussian signals injected. For each histogram with signal, the significance of the artificial bump is selected randomly in this interval.

- `number_of_samples` is the number of samples you want from your smoothed and function samples. We usually aim for between 100k (for prediction) and 1M (for training) total histograms. As an example, if you have 1000 functionnal bkg + 1000 MC smoothed bkg and you set this paramteers to 2, the resulting dataset will contain 4000 histograms.



### Train

The training step is used to train a model. It needs 2 datasets, a training one and validation one. The validation dataset should be ~10% the size of the training dataset. The training is controlled via this block of code.

```yaml
train:
  name: Znet3_DMfunc
  cuda_visible_devices: '0'
  output_dir: baseline/model_DMfunc
  train_dir: baseline/DM_and_func_bkg_with_gaussian_signal_train_seed_1_percent_with_signal_1.0
  val_dir: baseline/DM_and_func_bkg_with_gaussian_signal_val_seed_2_percent_with_signal_1.0
  resume_from_checkpoint: null
  batch_size: 5000
  num_epochs: 300
  min_delta: 1.0e-06
  rel_mse: false
  z_loss_wt: 1
  b_loss_wt: 1
  b_loss_type: mse
  learning_rate: 0.001
  lr_scheduler:
    CosineAnnealingLR:
      T_max: -1
      eta_min: 1.0e-06
      last_epoch: -1
      verbose: true
  num_workers: 4
  #comet:
  #  project_name: DDP
  csv_logger:
    name: BumpNet
  #device: 'gpu'
  device: 'cuda' # for ccin2p3
```

- `name` is the name you want to give to your trained model.

- `cuda_visible_devices`, `resume_from_checkpoint`, `batch_size`, `num_epochs`, `min_delta`, `rel_mse`, `learning_rate`, `lr_scheduler`, and `num_worker` are parameters of the training. They usually aren't changed (we take them to be optimzed), unless we want to try specifically optimizing one of them. 

- `output_dir` is the directory in which you want to place your trained model. It does not have to already exist.

- `train_dir` is the directory in which your training data set is. It is data created with the generate step.

- `val_dir` is the directory in which the validation data set is. It is generally created the same way as the training data, but with a different random seed.

- `z_loss_wt` and `b_loss_wt` are the weight/importance you want to give to z prediction and background prediction in the training. Per example, setting `b_loss_wt` to 0 and `z_loss_wt` to 1 would train a model only to predict significance, not predicting the background.

- The comet `project_name` is the name of the https://www.comet.com/ directory in which you want the loss plots to show for this training.

NOTE : In a linux-based cluster, you home directory should contain a hidden file called `.bashrc` in which you have to write

```bash
export COMET_API_KEY="<your_comet_api_key>"
export COMET_WORKSPACE="<your_comet_username>"
```
for comet to work for you.

- Alternatively, one can use a local `csv_logger`, which saves the loss values locally in a csv file. You should have either `comet` or `csv_logger` in the config file to enable logging, and if the config has both it will default to using comet.

- `device` selects the device you will use in your system: `'gpu'`, `'cuda'`, or `'cpu'` [among others](https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator) can be used.

### Predict

The prediction step uses a train model to predict on a dataset. The prediction is controlled via this block of code

```yaml
predict:
  checkpoint_path: baseline/model_DMfunc/Znet3_DMfunc/best_model.ckpt
  seed: 42
  batch_size: 5000
  num_workers: 4
  input_dirs: [
    "baseline/DM_and_func_bkg_with_gaussian_signal_test_seed_3_percent_with_signal_1.0", 
    "baseline/DM_and_func_bkg_with_gaussian_signal_test_seed_3_percent_with_signal_0.0", 
    "baseline/DM_bkg_with_gaussian_signal_test_seed_3_percent_with_signal_1.0", 
    "baseline/DM_bkg_with_gaussian_signal_test_seed_3_percent_with_signal_0.0", 
    "baseline/function_bkg_with_gaussian_signal_test_seed_3_percent_with_signal_1.0", 
    "baseline/function_bkg_with_gaussian_signal_test_seed_3_percent_with_signal_0.0", 
    "baseline/dir-with-root-file"
    ]
  output_dirs: [
    "baseline/prediction_DMfunc_to_DMfunc_sig", 
    "baseline/prediction_DMfunc_to_DMfunc_bkg", 
    "baseline/prediction_DMfunc_to_DM_sig", 
    "baseline/prediction_DMfunc_to_DM_bkg", 
    "baseline/prediction_DMfunc_to_func_sig", 
    "baseline/prediction_DMfunc_to_func_bkg", 
    "baseline/prediction_raw"
    ]
```

- `checkpoint_path` is the path to the trained model.

- `batch_size` and `num_workers` are machine learning parameters that are usually left untouched.

- `input_dirs` and `output_dirs` are two lists which need to have the same length are where the i-th element of one lists belong together. 

### Merge several BN output, to improve the precision

Assuming you have trained and run several BN's, you can merge their outputs, so you can improve the performances:
```yaml
mergeBNoutput:
  input_dirs: [
    "baseline/prediction_42/percent_with_signal_1.0",
    "baseline/prediction_43/percent_with_signal_1.0",
    "baseline/prediction_44/percent_with_signal_1.0",
    ]
  output_dir: baseline/prediction/
```


### Produce plot to validate the merging of BN outputs

To produce the plots that validate the merging step, one needs to provide:

- `input_dir_B` and `input_dir_S` : the directory names for 0% and 100% signal sample, containing the truth

- `predictions_merge_B` and `predictions_merge_S` : the directory names for 0% and 100% signal sample, for the merged predictions

- `predictions_B` and `predictions_S`: list of single predictions, for 0% and 100% signal samples


```yaml
validateMergingBN:
  input_dir_B: baseline/generated/test/percent_with_signal_0.0
  input_dir_S: baseline/generated/test/percent_with_signal_1.0
  predictions_B: 
  - baseline/prediction_42/percent_with_signal_0.0
  - baseline/prediction_43/percent_with_signal_0.0
  - baseline/prediction_44/percent_with_signal_0.0
  predictions_S: 
  - baseline/prediction_42/percent_with_signal_1.0
  - baseline/prediction_43/percent_with_signal_1.0
  - baseline/prediction_44/percent_with_signal_1.0
  predictions_merge_B: baseline/prediction/percent_with_signal_0.0
  predictions_merge_S: baseline/prediction/percent_with_signal_1.0
  output_dir: baseline/prediction_plot/
  format: png
```


### Plot

This step is used to generate plots of the results of your model according to it's predictions. The plotting/visualisation of the results can be controlled via this block of code

```yaml
plot:
  datasets:
  - output_dir: baseline/plots_DMfunc_to_DMfunc
    sig_input_dir: baseline/DM_and_func_bkg_with_gaussian_signal_test_seed_3_percent_with_signal_1.0
    sig_prediction_dir: baseline/prediction_DMfunc_to_DMfunc_sig
    bkg_input_dir: baseline/DM_and_func_bkg_with_gaussian_signal_seed_3_percent_with_signal_0.0
    bkg_prediction_dir: baseline/prediction_DMfunc_to_DMfunc_bkg
  - output_dir: baseline/plots_DMfunc_to_DM
    sig_input_dir: baseline/DM_bkg_with_gaussian_signal_test_seed_3_percent_with_signal_1.0
    sig_prediction_dir: baseline/prediction_DMfunc_to_DM_sig
    bkg_input_dir: baseline/DM_bkg_with_gaussian_signal_seed_3_percent_with_signal_0.0
    bkg_prediction_dir: baseline/prediction_DMfunc_to_DM_bkg
  - output_dir: baseline/plots_DMfunc_to_func
    sig_input_dir: baseline/function_bkg_with_gaussian_signal_test_seed_3_percent_with_signal_1.0
    sig_prediction_dir: baseline/prediction_DMfunc_to_func_sig
    bkg_input_dir: baseline/function_bkg_with_gaussian_signal_seed_3_percent_with_signal_0.0
    bkg_prediction_dir: baseline/prediction_DMfunc_to_func_bkg
  what_to_plot:
  - roc_curve
  - z_distributions
  - z_extremum
  - zmax_distributions
  - examples
  conditions:
    n_examples: 10
    signal_examples_condition: perf_sig.z_pred_max[i] >= 5
    background_examples_condition: perf_bkg.z_pred_max[i] >= 5
    deltaz_lim:
    - -5
    - 5
    deltazmax_lim:
    - -5
    - 5
    interpolate: false
    outliers_zmax_threshold: 5
    outliers_in_std_units: false
  format: pdf
  seed: 10
  max_rows: 10000
  verbose: true
  shuffle: true
  edge:
  - 0.1
  - 1
  show: false
  plot_std_dist: true
  plot_kde: true
```
- You put the samples that you want to plot in the `plot_file_configs` section. For each sample, you have to provide the directory in which the given sample is in `input_file_dir`, the directory in which the prediction made by DDP for this sample is in `prediction_dir`, and the directory in which the plots you want to go in `output_dir`.

NOTE : if you want to predict and plot at the same time (which is usually the case), the `input_file_dir` and the `prediction_dir` will be the same in both sections.

- In `what_to_plot` you ask the various plots that you want. All the possible plots are in the block of code above, just remove one to not plot it. A possible option is also `all`. All possible plots can be found in `plotting/plotting_options.txt`.

- In `conditions`, you can set various conditions for the "examples" that you asked to plot. There are many possible conditions. Notably, "n_examples" sets the number of examples that you want to plot.

- `seed` is the random seed associated with the selection of the `hist_with_zpl` and `hist_with_prediction` (selected randomely amongst all of the sample's histogram).


## Automisation using Snakemake

The ci pipeline has been automated using snakemake for testing purposes in R&D (right now it only tests functions). The pipeline can be lauched using: 
```
snakemake --profile BumpNet/snakemake/workflow/profiles/<cluster> --snakefile BumpNet/snakemake/workflow/ci_pipeline.smk 

```
The profile is individual to each cluster and can be configured. The currect implementation expects an already existing conda environment (see above) but will soon be replaced by a universal container image.

To stop snakemake from regenrating an expensive file, just `touch` the file (only if you are *really* sure nothing changed). If snakemake is supposed to run some lengthy jobs, it is advised to use `screen` or `tmux` to safely disconnect from the cluster. 

### Installation of Snakemake
Snakemake is best installed in its own conda environment
```
conda create --name snakemake python=3.12
conda activate snakemake

pip install pulp==2.6.0
pip install snakemake
pip install snakemake-executor-plugin-slurm # if slurm is used
```
## Enjoy !! 
