FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

LABEL maintainer="Manuel"
LABEL description="Self-Supervised Learning Framework for Medical Image Representation Learning"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories for data and checkpoints
RUN mkdir -p /data /checkpoints /results

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=8

# Default entrypoint: pretraining
ENTRYPOINT ["python"]
CMD ["scripts/pretrain.py", "--config", "configs/dino_mammography.yaml"]
