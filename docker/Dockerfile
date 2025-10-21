FROM rootproject/root:latest

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y python3-venv git

# Create virtual environment in /opt
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variable for rpy2 build
ENV RPY2_CFFI_MODE=ABI

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Register Jupyter kernel
RUN python -m ipykernel install --name root-env --display-name "ROOT Python"