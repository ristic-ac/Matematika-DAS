FROM python:3.12-slim-bookworm

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    swi-prolog swi-prolog-nox \
    git \
    build-essential libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pyswip

# Install Aleph (non-interactively)
RUN swipl -g "pack_install(aleph, [interactive(false)])" -t halt

# Ensure the directories exist in the image
RUN mkdir -p /app/scripts /app/outputs

RUN pip install xgboost

RUN pip install --no-cache-dir pyarrow
# Note: To bind the host's ./scripts folder (where the Dockerfile is) to /app/scripts in the container,
# run the container with:
#   docker run -v $(pwd)/scripts:/app/scripts -v $(pwd)/output:/app/output ...

# Default run command (change as needed)
ENTRYPOINT ["/bin/bash"]
