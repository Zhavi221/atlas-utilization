FROM rootproject/root:latest

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y python3-venv git

# Create virtual environment in /opt
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    ipykernel \
    uproot \
    awkward \
    vector \
    matplotlib \
    tqdm \
    pandas \
    notebook \
    atlasopenmagic \
    numpy \
    setuptools \
    xrootd \
    fsspec-xrootd \
    cernopendata_client \
    jupyterlab \
    debugpy  # For VS Code debugging

# Register Jupyter kernel
RUN python -m ipykernel install --name root-env --display-name "ROOT Python"

