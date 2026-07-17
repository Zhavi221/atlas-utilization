# Tutorial BumpNet

## Introduction

Welcome to the tutorial of the BumpNet Framework! The idea of BumpNet is to train a network based on *shapes* of smoothly falling background and signal bumps in simulation. Later, this network can be applied to experimental data to identify signal. Please also consult the `README.md` which hosts all details about the framework.

### 1. Smoothing 

As a part of the workflow, the first step is to generate the smoothly falling background shapes. For that, we use histograms filled with simulated data of the physics processes we are interestedi. Because these are not smooth, but have statistical fluctuations, we apply the smoothing procedure. The smoothing is based on fitting polynominal functions or using gaussian processed in order to estimate the underlying PDF of the histogram.

### 2. Functions

After the smoothing, we obtain a range in which the smoothed histograms live. Based on that range, we create histograms from analytical function we emperically know can describe these smoothly falling backgrounds. We do that in order to diversify the trainings dataset. 

### 3. Generate

Based on these PDF shapes, we create artifically data-like histograms by poisson fluctuate the bin content. This way, we also generate a way larger dataset for training than just our base histograms. In this step, also the labels are being added which are the per bin likelihood ratio significance with respect to the underlying PDF. In some histograms, gaussian signal shapes are being inserted. We have to generate a statistics high trainings dataset and a testing and validation dataset with smaller statistics. Because that takes a long time, this process is parallelised.

### 4. Training

In the training step, we train our BumpNet model with our training dataset we created before. BumpNet predicts a per bin significant and is trained to match the true significance as given by the likelihoood ratio as close as possible. Using the validation dataset, we monitor that the model does not overfit on the trainings dataset. 

### 5. Prediction

After the training, we apply BumpNet to our dedicated testing dataset so that we can evaluate its performance. 

### 6. Plotting

Here we gather everything and evaluate multiple performance metrics of BumpNet. 


## Setup

### Conda Setup

To run the BumpNet workflow, you need an installation of snakemake, best done in an conda environment. If you do not have conda installed, please do so by calling:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Now add this in your `.bashrc`
```
export PATH=~/miniconda/bin:$PATH
```
Open a new shell session and validate that everything works as expected. 

### Snakemake Setup

```
conda create --name snakemake python=3.12
conda activate snakemake

pip install pulp==2.6.0
pip install snakemake
pip install snakemake-executor-plugin-slurm # if slurm is used
```

## Execution

As introduced earlier, snakemake is a tool to automise the workflow of the analysis. That means, you only need to run one command and all steps of the analysis will be run. To start, you best open a `tmux` screen via
```
tmux new -s BumpNet
```
in order to run the command and be able to disconnect from the ssh server via aborting the command. 
Inside the tmux screen, you load the snakemake conda environment via
```
conda activate snakemake
```
Next, you can run the tutorial via 
```
snakemake --profile BumpNet/snakemake/workflow/profiles/<cluster> --snakefile BumpNet/snakemake/workflow/tutorial.smk
```
from the directory in the level above the BumpNet directory. As the cluster you have to fill in the respective facility you are working in, e.g. geneva.
After launching the snakemake command, you can deconnect from the tmux via `CTRL + B D`. and reconnect via `tmux a -t BumpNet`. Now watch and monitor your workflow until it either crashes or suceeeds :) 

