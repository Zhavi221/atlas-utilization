configfile: "configs/snakemake.yaml"

container: config["container"]


rule all:
    input:
        "test/plots"


rule function:
    output:
        directory("test/function_bkg")
    shell:
        r"""
        python3 sample_generation/function.py --config configs/ci_pipeline.yaml
        """


rule smooth:
    input: 
        "input/ci/rebinned_NOSYS.root"
    output:
        directory("test/MC_bkg_smoothed")
    threads: 4
    shell:
        r"""
        python3 sample_generation/smooth.py --config configs/ci_pipeline.yaml
        """


rule generate:
    input: 
        func_path = "test/function_bkg",
    output:
        directory("test/generated/{split}/percent_with_signal_{percent_with_signal}")
    shell:
        r"""
        python sample_generation/generate.py --config configs/ci_pipeline.yaml --input_paths {input.func_path} --output_dir {output} --percent_with_signal {wildcards.percent_with_signal} 
        """

rule train:
    input: 
        "test/generated/train/percent_with_signal_1.0",
        "test/generated/val/percent_with_signal_1.0"
    output:
        "test/model_func/Znet3_func/best_model.ckpt"
    shell:
        r"""
        python3 ml_scripts/train.py --config configs/ci_pipeline.yaml
        """

rule predict:
    input: 
        input_dir = "test/generated/test/percent_with_signal_{percent_with_signal}",
        ckpt = "test/model_func/Znet3_func/best_model.ckpt"
    output:
        directory("test/prediction_seed1_{percent_with_signal}")
    shell:
        r"""
        python3 ml_scripts/predict.py --config configs/ci_pipeline.yaml --input_dirs {input.input_dir} --output_dirs {output}  --checkpoint_path {input.ckpt}
        """

rule mergeBN:
    input:
        predictions=["test/prediction_seed1_{percent_with_signal}",]
    output:
        directory("test/predictionMerged_{percent_with_signal}")
    shell:
        r"""
        python3 ml_scripts/mergeBNoutput.py --config configs/analysis.yaml --output_dir {output} --input_dirs {input.predictions}
        """

rule validate_mergeBN:
    input:
        predictions_S=["test/prediction_seed1_percent_with_signal_1.0",],
        predictions_B=["test/prediction_seed1_percent_with_signal_0.0",],
        predictions_merge_S="test/predictionMerged_1.0",
        predictions_merge_B="test/predictionMerged_0.0",
        input_dir_S="test/generated/test/percent_with_signal_1.0",
        input_dir_B="test/generated/test/percent_with_signal_0.0"
    output:
        directory(f"test/validateMerging/generated/")
    shell:
        r"""
        python3 plotting/validate_merging_BN.py --config configs/analysis.yaml --output_dir {output} \
            --predictions_S {input.predictions_S} --predictions_merge_S {input.predictions_merge_S} \
            --predictions_B {input.predictions_B} --predictions_merge_B {input.predictions_merge_B} \
            --input_dir_S {input.input_dir_S} --input_dir_B {input.input_dir_B}
        """

rule plot:
    input: 
        sig_input_dir = "test/generated/test/percent_with_signal_1.0",
        bkg_input_dir = "test/generated/test/percent_with_signal_0.0",
        sig_pred_dir = "test/predictionMerged_1.0",
        bkg_pred_dir = "test/predictionMerged_0.0"
    output:
        directory("test/plots")
    shell:
        r"""
        python3 plotting/plot.py --config configs/ci_pipeline.yaml --bkg_input_dir {input.bkg_input_dir} --bkg_prediction_dir {input.bkg_pred_dir} --sig_input_dir {input.sig_input_dir} --sig_prediction_dir {input.sig_pred_dir} --output_dir {output}
        """
