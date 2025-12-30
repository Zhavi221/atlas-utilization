This guide explains the different methods of running the job in WIS. It might include some things that exist in the tutorial or the main README file, and it might include some information that is useful on other clusters. However, it is written and directed for people in WIS. It mostly explains how to run without snakemake, because Ilan hasn't tried using it yet.

In WIS when running on wipp we have two main options:
1. running locally (in the zsh/bash terminal)
2. running on PBS nodes

additionally, there are two main modes of running:
1. with conda environment
2. with apptainer and containers.

It is suggested that you run with apptainer, to make the process as reproducible as possible. For the sake of simplicity I will explain how to run with conda environment locally first, then how to run with apptainer, and finally how to run on PBS nodes.

# Do this for any way you choose in any case:

```
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
```

this will initialize your conda environments

## Running Locally with Conda Environment

First, initialize the `common` environment:
```
conda activate common
```

Now, to run locally with `DDP.py` just run the following:
```
python DDP.py --config config.yaml
```
where `config.yaml` is the config file you will use for running. Make sure that any path you use is either relative to the directory you are in, or ideally with an absolute path.

Note: on wipp-an1/2 we do not have a GPU, and you should use at most 4 CPUs in parallel. In fact, it's better if you use only 1 at any point in time.

## Running Locally with apptainer

Initialize the apptainer environment:
```
conda activate apptainer
```

Now to run with the container run this:
```
apptainer exec --cleanenv --nv -B base_dir container.sif python DDP.py --config config.yaml
```
where:
- `--cleanenv` is needed so apptainer uses only the container for running
- `--nv` uses nvidia, so I guess it's to be able to run training (although maybe it's possible without this, I haven't tried)
- `-B base_dir` is the base dir to mount on apptainer. This should be the directory which includes all the files that will be needed for running: `DDP.py`, the config file, and all files referenced by the config. Note that any path given cannot have a symbolic link instead of a path. So, if for example you have a symbolic path `~/sympath` which leads to you storage directory `/storage/agrp/user`, you have to write `/storage/agrp/user/dir` and not `~/sympath/dir`. This is a better practice in general (and of course I usually don't listen to the better practice)
- `container.sif` the container SIF file that contains all the packaging information needed to run
- `python DDP.py --config config.yaml` the command you run on apptainer.

## Running on PBS nodes

First, let's show how to run without apptainer. We need a .sh file to run, and a good example is `run_on_node_pbs.sh`. It has several lines:
```
#PBS -m n

source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/conda/24.5.0u/envs/common/lib

cd ${CWD}

python DDP.py --config ${CONFIG}
```
I'll go over each line:
- `#PBS -m n` this prevents sending email in the nodes. This is a requirement by LCG managers. If you don't have it nothing bad will happen, but it's better to keep it in case you accidentally have a command that sends an email. Why it's bad to send an email from a PBS node? I don't know. I guess you can ask the LCG managers.
- `source ...; conda activate common` makes sure the correct environment is activated on the pbs node itself.
- `export LD_LIBRARY_PATH=...` makes sure the correct library is used for running.
- `cd ${CWD}` changes directory to where the `DDP.py` file is. The `CWD` variable is given to the node in the `qsub` command, which will be explained later.
- `python ...` is the normal python command.

To send the .sh script to the node, we use the `qsub` command with several options:
```
qsub \
    -o stdout.out \
    -e stderr.err \
    -q N \
    -N job_name \
    -l walltime=12:00:00,mem=50gb,ncpus=1,ngpus=0,io=1 \
    -v CONFIG=config.yaml,CWD=base_dir_with_python_code \
    run_on_node_pbs.sh
```
explanation on each line:
- `-o stdout.out; -e stderr.err` creates files in which the stdout and stderr will be stored. You should provide the full path (so `/storage/...`)
- `-q N` which queue to use. The `N` queue is the normal, and it selects itself to which queue to send the job, mostly based on the walltime.
- `-N job_name` the name the job will have. This is mostly to find it more easily on the `qtop`. It can have any name, or no name if you prefer.
- `-l <options>` this is the most important path: state the resources you need. Each resource is pretty self explanatory, but to be thorough:
    - `walltime=HH:MM:SS` is how much time you need to run the job. If you run for more than the walltime, the job will stop. For a job with no gpus, depending on how much walltime you ask for, you will get a different node:
        - for up to 2 hours of walltime you will get the short job nodes
        - for more than 2 hours and up to 12 hours you will get the medium job nodes
        - for more than 12 hours and up to 72 hours you will get the long job nodes
    Ideally, you should measure how long a single job should take, and ask for walltime of not more than that. In reality, you can ask for 2, 12, or 72 hours if your job will take $T\leq 2h$, $2h<T\leq 12h$, or $12h<T<72h$ respectly.
    - `mem=XXgb` how much RAM memory you need. Generally the smaller the better, but know that all non-gpu nodes have at least 247gb of memory, and one even has 1tb of memory, so you can safely ask for more RAM than you'd think a normal person would need. I usually ask for 50gb when I load large files, or 32gb when they are smaller. That should be more than enough.
    - `ncpus=N` how many cpu cores you will use. This is important: **only ask for the number of nodes you actually need**. This blocks the cpus for other users, so if you ask for 36 nodes but actually use 1, you are wasting resources. The other thing is: if you ask for 1 cpu but your program uses more than one (for example with multiprocessing pool of 2 or more workers), it will work but it will cause problems for other jobs that are also running on the node, and you will get an angry email from the lcg managers.
    There is of course a maximal number of nodes you can ask for. All nodes have a minimum of 36 cores, and there are many nodes with 52 cores. There is one node with 168 cores and two more with 192 cores each (all three have 1tb RAM each). These are only for gigantic jobs, so generally no need for them. The gpu nodes have either 48 cores or 64 cores.
    - `ngpus=N` how many gpus you need. Generally only one is needed per training. All gpu nodes have at least 2 gpus, some have 4, but there really shouldn't be a need for using more than 1 gpu at any time.
    - `io=N` how much reading/writing from/to disk/web will be done per second, in units of mega bytes per second. It's hard to tell how much will be needed, generally the way to know is to take the amount of reading/writing that will be done (so if you read 1gb of files, that is 1gb), divide it by the walltime you ask for, and round up in some manner to be sure you have enough. For example, say I read 1gb of files and write 1gb of files, so 2gb in total, and run the job with a walltime of 2hr; this means an average of ~0.14MB/s. So, io=1 should be enough. I think there is a limit for the io you can ask for, and it is of course also limited by the capabilities of the hardware itself, but I don't know them. I have been using io=1 for most of my jobs without any issue though, so maybe don't go over 5? You can always contact lcg managers to ask them.
You can always check out all the nodes capabilities with `pbsnodes -aS`. For more options when running pbs, you can look for many documentations online. [this](https://help.altair.com/2022.1.0/PBS%20Professional/PBSUserGuide2022.1.pdf) is the official documentation of the professional version, but there are simpler guides online like [this one](https://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm).

The file `submit_pbs.py` was created to facilitate sending many jobs automatically to the pbs cluster, but it is really outdated. You should make automation scripts on your own as you go. I will try to make an updated version at some point.

When running an apptainer job on the pbs nodes, you can use the `submit_wipp_apptainer.sh` job, just note that it has different input variables than `run_on_node_pbs.sh`.

Final note: to view your running jobs, use `qtop`. It should be straightforward to understand what each column means, but if it isn't - ask around :)