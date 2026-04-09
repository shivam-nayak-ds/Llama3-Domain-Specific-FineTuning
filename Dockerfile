# Use NVIDIA CUDA devel image for GPU support and compilation (nvcc)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy requirements and project files
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Unsloth and specific dependencies for inference
RUN pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip3 install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Copy the application code and assets
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY streamlit_app.py .
COPY app.py .
COPY main.py .
COPY .env.example .env

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Set the command to run Streamlit app by default
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
