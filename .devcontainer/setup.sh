#!/bin/bash

# Update package lists
apt-get update

# Install necessary packages
apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment
python3 -m venv /workspace/venv

# Install Python packages in the virtual environment
/workspace/venv/bin/pip install ipykernel uproot awkward vector matplotlib tqdm pandas numpy

# Register the virtual environment as a Jupyter kernel
/workspace/venv/bin/python -m ipykernel install --user --name=workspace_venv