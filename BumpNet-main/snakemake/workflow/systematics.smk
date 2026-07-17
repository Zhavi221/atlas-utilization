configfile: "BumpNet/configs/snakemake.yaml"

container: config["container"]

systematics = config["sys"]


rule all:
    input:
        expand("ANA-EXOT-2023-18/analysis2.0/plots/BumpNet/{mode}/{sys}", sys=systematics, mode=["smoothed", "generated"])


rule smooth:
    input:
        input_path="ANA-EXOT-2023-18/analysis2.0/rebinned/{sys}/rebinned.root"
    output:
        directory("ANA-EXOT-2023-18/analysis2.0/smoothed/{sys}")
    shell:
        r"""
        python BumpNet/sample_generation/smooth.py --config BumpNet/configs/systematics.yaml --input_path {input.input_path} --output_dir {output}
        """

rule generate:
    input:
        input_path="ANA-EXOT-2023-18/analysis2.0/smoothed/{sys}"
    output:
        directory("ANA-EXOT-2023-18/analysis2.0/generated/{sys}")
    shell:
        r"""
        python BumpNet/sample_generation/generate.py --config BumpNet/configs/systematics.yaml --input_paths {input.input_path} --output_dir {output}
        """

rule predict:
    input: 
        input_dir="ANA-EXOT-2023-18/analysis2.0/{mode}/{sys}",
        ckpt="BumpNet/BumpNet_models/250805_altasmc_functions/best_model.ckpt"
    output:
        directory("ANA-EXOT-2023-18/analysis2.0/prediction/{mode}/{sys}")
    shell:
        r"""
        python3 BumpNet/ml_scripts/predict.py --config BumpNet/configs/systematics.yaml --input_dirs {input.input_dir} --output_dirs {output}
        """

rule plot:
    input: 
        prediction_dir="ANA-EXOT-2023-18/analysis2.0/prediction/{mode}/{sys}",
        input_dir="ANA-EXOT-2023-18/analysis2.0/{mode}/{sys}",
    output:
        directory("ANA-EXOT-2023-18/analysis2.0/plots/BumpNet/{mode}/{sys}")
    shell:
        r"""
        python3 BumpNet/plotting/plot.py --config BumpNet/configs/systematics.yaml --bkg_input_dir {input.input_dir} --bkg_prediction_dir {input.prediction_dir} --output_dir {output}
        """
