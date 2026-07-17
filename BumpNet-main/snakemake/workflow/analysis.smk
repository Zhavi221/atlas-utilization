import random
import os

configfile: "BumpNet/configs/snakemake.yaml"

# Load configuration from snakemake config file
container: config["container"]

n_parallel = config["n_parallel"]
n_BN = config["n_BN"]

train_size = config["train_size"]

base_dir = config["directory"]
os.makedirs(base_dir, exist_ok=True)

systematics = config["sys"]

split_shapes = config["split_shapes"]

mode_smoothing = config["mode_smoothing"]

# create three independent RNGs, seeded differently
rng1 = random.Random(12345)
rng2 = random.Random(67890)
rng3 = random.Random(13579)
rng4 = random.Random(78782)

# generate three independent sequences of seeds
seq1 = [rng1.randint(0, 1_000_000_000) for _ in range(n_parallel)]
seq2 = [rng2.randint(0, 1_000_000_000) for _ in range(n_parallel//10)]
seq3 = [rng3.randint(0, 1_000_000_000) for _ in range(n_parallel//10)]
seq4 = [rng4.randint(0, 1_000_000_000) for _ in range(n_BN)]

# Set test/val sample size to be 10% of train size via number of parallel jobs
SPLIT_SEEDS = {
    "train": seq1,
    "val":   seq2,
    "test":  seq3
}


rule all:
    input:
        f"{base_dir}/plots/BumpNet/generated",
        f"{base_dir}/validateMerging/generated/",
        expand(f"{base_dir}/plots/BumpNet/{{mode}}/{{sys}}", sys=systematics, mode=mode_smoothing)


rule smooth:
    input:
        input_path=f"{base_dir}/rebinned/{{sys}}/rebinned.root",
        nominal_path = lambda wildcards: (
            [] if wildcards.sys == "NOSYS"
            else f"{base_dir}/smoothed/NOSYS"
        )
    output:
        directory(f"{base_dir}/smoothed/{{sys}}")
    shell:
        r"""
        python BumpNet/sample_generation/smooth.py --config BumpNet/configs/analysis.yaml --input_path {input.input_path} --output_dir {output} --nominal_dir {input.nominal_path}
        """

rule function:
    input:
        smooth_dir=f"{base_dir}/smoothed/NOSYS"
    output:
        directory(f"{base_dir}/functions")
    shell:
        r"""
        python3 BumpNet/sample_generation/function.py --config BumpNet/configs/analysis.yaml --smooth_dir {input} --output_dir {output}
        """

if split_shapes:
    rule split:
        input: 
            MC_path=f"{base_dir}/smoothed/NOSYS",
            func_path=f"{base_dir}/functions"
        output:
            directory(f"{base_dir}/bkg_split/test/NOSYS"),
            directory(f"{base_dir}/bkg_split/test/functions"),
            directory(f"{base_dir}/bkg_split/train/NOSYS"),
            directory(f"{base_dir}/bkg_split/train/functions"),
            directory(f"{base_dir}/bkg_split/val/NOSYS"),
            directory(f"{base_dir}/bkg_split/val/functions")
        shell:
            r"""
            python3 BumpNet/sample_generation/split.py --config BumpNet/configs/analysis.yaml --input_paths {input.MC_path} {input.func_path}
            """

rule generate:
    input:
        MC_path=f"{base_dir}/bkg_split/{{split}}/NOSYS" if split_shapes else f"{base_dir}/smoothed/NOSYS",
        func_path=f"{base_dir}/bkg_split/{{split}}/functions" if split_shapes else f"{base_dir}/functions"
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
        f"{base_dir}/BumpNetModel_{{BNseed}}/best_model.ckpt"
    shell:
        r"""
        python BumpNet/ml_scripts/train.py --config BumpNet/configs/analysis.yaml --train_dir {input.train_path} --val_dir {input.val_path} --output_dir {base_dir} --seed {wildcards.BNseed}
        """

rule predict_generated:
    input: 
        input_dir=f"{base_dir}/merged/test/percent_with_signal_{{percent_with_signal}}",
        ckpt=f"{base_dir}/BumpNetModel_{{BNseed}}/best_model.ckpt"
    output:
        directory(f"{base_dir}/prediction_{{BNseed}}/percent_with_signal_{{percent_with_signal}}")
    shell:
        r"""
        python3 BumpNet/ml_scripts/predict.py --config BumpNet/configs/analysis.yaml --input_dirs {input.input_dir} --output_dirs {output} --checkpoint_path {input.ckpt}
        """

rule mergeBN_generated:
    input: 
        predictions=lambda wc: expand(
            f"{base_dir}/prediction_{{BNseed}}/percent_with_signal_{{percent_with_signal}}",
            percent_with_signal=wc.percent_with_signal,
            BNseed=seq4
        )
    output:
        directory(f"{base_dir}/prediction/percent_with_signal_{{percent_with_signal}}"),
    shell:
        r"""
        python3 BumpNet/ml_scripts/mergeBNoutput.py --config BumpNet/configs/analysis.yaml --output_dir {output} --input_dirs {input.predictions}
        """

rule validate_mergeBN_generated:
    input: 
        predictions_S=lambda wc: expand(
            f"{base_dir}/prediction_{{BNseed}}/percent_with_signal_1.0",
            BNseed=seq4
        ),
        predictions_B=lambda wc: expand(
            f"{base_dir}/prediction_{{BNseed}}/percent_with_signal_0.0",
            BNseed=seq4
        ),
        predictions_merge_S=f"{base_dir}/prediction/percent_with_signal_1.0",
        predictions_merge_B=f"{base_dir}/prediction/percent_with_signal_0.0",
        input_dir_S=f"{base_dir}/merged/test/percent_with_signal_1.0",
        input_dir_B=f"{base_dir}/merged/test/percent_with_signal_0.0"
    output:
        directory(f"{base_dir}/validateMerging/generated/")
    shell:
        r"""
        python3 BumpNet/plotting/validate_merging_BN.py --config BumpNet/configs/analysis.yaml --output_dir {output} \
            --predictions_S {input.predictions_S} --predictions_merge_S {input.predictions_merge_S} \
            --predictions_B {input.predictions_B} --predictions_merge_B {input.predictions_merge_B} \
            --input_dir_S {input.input_dir_S} --input_dir_B {input.input_dir_B} 
        """

rule plot_generated:
    input: 
        val=f"{base_dir}/validateMerging/generated/",
        sig_input_dir=f"{base_dir}/merged/test/percent_with_signal_1.0",
        bkg_input_dir=f"{base_dir}/merged/test/percent_with_signal_0.0",
        sig_pred_dir=f"{base_dir}/prediction/percent_with_signal_1.0",
        bkg_pred_dir=f"{base_dir}/prediction/percent_with_signal_0.0"
    output:
        directory(f"{base_dir}/plots/BumpNet/generated/")
    shell:
        r"""
        python3 BumpNet/plotting/plot.py --config BumpNet/configs/analysis.yaml --bkg_input_dir {input.bkg_input_dir} --bkg_prediction_dir {input.bkg_pred_dir} --sig_input_dir {input.sig_input_dir} --sig_prediction_dir {input.sig_pred_dir} --output_dir {output}
        """

rule predict_raw:
    input: 
        input_dir=f"{base_dir}/smoothed/{{sys}}",
        ckpt=f"{base_dir}/BumpNetModel_{{BNseed}}/best_model.ckpt"
    output:
        directory(f"{base_dir}/prediction_{{BNseed}}/{{mode}}/{{sys}}")
    params:
        input_dir = lambda wildcards, input: (
            input.input_dir if wildcards.mode == "accepted" else f"{input.input_dir}/rejected"
        )
    shell:
        r"""
        python3 BumpNet/ml_scripts/predict.py --config BumpNet/configs/analysis.yaml --input_dirs {params.input_dir} --output_dirs {output} --checkpoint_path {input.ckpt}
        """

rule mergeBN_raw:
    input: 
        predictions=lambda wc: expand(
            f"{base_dir}/prediction_{{BNseed}}/{wc.mode}/{wc.sys}",
            BNseed=seq4
        )
    output:
        directory(f"{base_dir}/prediction/{{mode}}/{{sys}}")
    shell:
        r"""
        python3 BumpNet/ml_scripts/mergeBNoutput.py --config BumpNet/configs/analysis.yaml --output_dir {output} --input_dirs {input.predictions}
        """

rule plot_raw:
    input: 
        bkg_input_dir=f"{base_dir}/smoothed/{{sys}}",
        bkg_pred_dir=f"{base_dir}/prediction/{{mode}}/{{sys}}"
    output:
        directory(f"{base_dir}/plots/BumpNet/{{mode}}/{{sys}}")
    params:
        bkg_input_dir = lambda wildcards, input: (
            input.bkg_input_dir if wildcards.mode == "accepted" else f"{input.bkg_input_dir}/rejected"
        )
    shell:
        r"""
        python3 BumpNet/plotting/plot.py --config BumpNet/configs/analysis.yaml --bkg_input_dir {params.bkg_input_dir} --bkg_prediction_dir {input.bkg_pred_dir}  --output_dir {output}
        """


