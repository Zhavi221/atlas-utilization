* Tutorial: Znet3 with DM smoothed histograms, functions, and the new framwork
Dated: 2023-09-07
Full workflow for smoothing, functions, generating, training, predicting, and plotting, using Béluga.



** Setup **

If running interactively in the terminal:
```bash
# At login:
salloc --time=3:0:0 --nodes=1 --cpus-per-task=40 --mem=16000M --account=def-arguinj --gpus-per-node=1

# When submitting jobs
bash submit_rorqual.sh <config_file>
```

NOTE: if not training, better remove the gpu.



If submitting a batch job (a job that Béluga performs 'hidden'):
```bash
sbatch submit_rorqual.sh <config_file>
```

The configuration files are found in the configs directory, and configs/baseline_DMfunc.yaml is used as an example in this tutorial.

NOTE: you can check the status of a submitted job by seeing the last lines of the log file with the command `tail <log file>`. To check some stats about the job after it has finished, such as how long it took or how much memory it consumed, use the command `seff <job ID>`. You can also check the status of your jobs, including the interative ones, with the command `sq`.


If you have some predictions, data preparation, data analysis, or tests to conduct, you can use Rorqual's JupyterHub in a browser : https://jupyterhub.rorqual.computecanada.ca/hub/spawn . You don't need gpu unless you plan to train, you can ask for 10 cores and 30000 MB of memory. The interface JupyterLab is great, as you can access both a Jupyter Notebook and a terminal at the same time.

NOTE: sometimes, Rorqual won't allow a user to have a salloc and a JupyterLab allocation at the same time.










