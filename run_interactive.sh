bsub -m "cn650!" -q interactive -R "rusage[mem=8GB]" -n 4 -R "affinity[thread]" -R "span[hosts=1]" -Is /bin/bash -c "ml Singularity; singularity exec rootproject_latest.sif python main_pipeline.py"b

# qsub -I -q N -N im_job -l select=1:ncpus=4:mem=4000mb -l io=5