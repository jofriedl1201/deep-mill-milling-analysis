FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Install basics
RUN apt-get update && apt-get install -y git build-essential wget unzip

# Install Python deps
RUN pip install numpy yacs trimesh tqdm scikit-learn pandas

# Install O-CNN (Attempt pip install, if fails, we are stuck but usually works on Linux)
RUN pip install ocnn

# Set workdir
WORKDIR /workspace

# Clone DeepMill
RUN git clone https://github.com/fanchao98/DeepMill.git

# Setup structure
WORKDIR /workspace/DeepMill/projects

# Create Dummy Data Generation Script
COPY generate_dummy_data.py .

# Run Harness
CMD ["python", "generate_dummy_data.py"]
