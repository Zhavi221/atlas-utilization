import random
import os 


configfile: "BumpNet/configs/snakemake.yaml"

# Load configuration from snakemake config file
container: config["container"]

n_parallel = 10

train_size = 100_000

base_dir = "tutorial"
os.makedirs(base_dir, exist_ok=True)

# create three independent RNGs, seeded differently
rng1 = random.Random(12345)
rng2 = random.Random(67890)
rng3 = random.Random(13579)

# generate three independent sequences of seeds
seq1 = [rng1.randint(0, 1_000_000_000) for _ in range(n_parallel)]
seq2 = [rng2.randint(0, 1_000_000_000) for _ in range(n_parallel//10)]
seq3 = [rng3.randint(0, 1_000_000_000) for _ in range(n_parallel//10)]

# Set test/val sample size to be 10% of train size via number of parallel jobs
SPLIT_SEEDS = {
    "train": seq1,
    "val":   seq2,
    "test":  seq3
}


rule all:
    input:
        f"{base_dir}/plots/"


rule smooth:
    input:
        input_path="/srv/beegfs/scratch/groups/rodem/DarkMachines/backgrounds/background_chan2b_7.8/histos_Bmin30_Bmax200_thr0_massOver2_b.csv"
    output:
        directory(f"{base_dir}/smoothed")
    shell:
        r"""
        python BumpNet/sample_generation/smooth.py --config BumpNet/configs/analysis.yaml --input_path {input.input_path} --output_dir {output}
        """

rule function:
    input:
        smooth_dir=f"{base_dir}/smoothed"
    output:
        directory(f"{base_dir}/functions")
    shell:
        r"""
        python3 BumpNet/sample_generation/function.py --config BumpNet/configs/analysis.yaml --smooth_dir {input} --output_dir {output}
        """

rule generate:
    input:
        MC_path=f"{base_dir}/smoothed",
        func_path=f"{base_dir}/functions"
    output:
        directory(f"{base_dir}/generated/{{split}}/percent_with_signal_{{percent_with_signal}}/seed_{{seed}}")
    params:
        number_of_samples=lambda wildcards: f"{train_size//n_parallel}"
    shell:
        r"""
        python BumpNet/sample_generation/generate.py --config BumpNet/configs/analysis.yaml --input_paths {input.MC_path} {input.func_path} --output_dir {output} --percent_with_signal {wildcards.percent_with_signal} --number_of_samples {params.number_of_samples} --seed {wildcards.seed}
        """

rule merge:
    input:
        lambda wc: expand(
            f"{base_dir}/generated/{{split}}/percent_with_signal_{{percent_with_signal}}/seed_{{seed}}",
            split=wc.split,
            percent_with_signal=wc.percent_with_signal,
            seed=SPLIT_SEEDS[wc.split]
        )
    output:
        directory(f"{base_dir}/merged/{{split}}/percent_with_signal_{{percent_with_signal}}/")
    shell:
        r"""
        python BumpNet/sample_generation/merge.py --config BumpNet/configs/analysis.yaml --input_paths {input} --output_dir {output} 
        """

    

rule train:
    input:
        train_path=f"{base_dir}/merged/train/percent_with_signal_1.0",
        val_path=f"{base_dir}/merged/val/percent_with_signal_1.0"
    output:
        f"{base_dir}/BumpNetModel/best_model.ckpt"
    shell:
        r"""
        python BumpNet/ml_scripts/train.py --config BumpNet/configs/analysis.yaml --train_dir {input.train_path} --val_dir {input.val_path} --output_dir {base_dir}
        """

rule predict:
    input: 
        input_dir=f"{base_dir}/merged/test/percent_with_signal_{{percent_with_signal}}",
        ckpt=f"{base_dir}/BumpNetModel/best_model.ckpt"
    output:
        directory(f"{base_dir}/prediction/percent_with_signal_{{percent_with_signal}}")
    shell:
        r"""
        python3 BumpNet/ml_scripts/predict.py --config BumpNet/configs/analysis.yaml --input_dirs {input.input_dir} --ckpt {input.ckpt} --output_dirs {output}
        """

rule plot:
    input: 
        sig_input_dir=f"{base_dir}/merged/test/percent_with_signal_1.0",
        bkg_input_dir=f"{base_dir}/merged/test/percent_with_signal_0.0",
        sig_pred_dir=f"{base_dir}/prediction/percent_with_signal_1.0",
        bkg_pred_dir=f"{base_dir}/prediction/percent_with_signal_0.0"
    output:
        directory(f"{base_dir}/plots/")
    shell:
        r"""
        python3 BumpNet/plotting/plot.py --config BumpNet/configs/analysis.yaml --bkg_input_dir {input.bkg_input_dir} --bkg_prediction_dir {input.bkg_pred_dir} --sig_input_dir {input.sig_input_dir} --sig_prediction_dir {input.sig_pred_dir} --output_dir {output}
        """

