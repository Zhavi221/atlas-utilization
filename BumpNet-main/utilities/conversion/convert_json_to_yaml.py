import json
import yaml
import argparse
from pathlib import Path

def convert_json_to_yaml(input_path, output_path=None):
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")

    # Default output path
    if output_path is None:
        output_path = input_path.with_suffix(".yaml")
    else:
        output_path = Path(output_path)

    # Read JSON
    with open(input_path, "r") as f:
        data = json.load(f)

    # Write YAML
    with open(output_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Converted '{input_path}' → '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON config to YAML")
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument("-o", "--output_yaml", help="Optional output YAML file path")
    args = parser.parse_args()

    convert_json_to_yaml(args.input_json, args.output_yaml)

