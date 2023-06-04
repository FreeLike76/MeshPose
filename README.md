# MeshPose
A python package for visual localization.

## Pre-requisites
Conda environment with python 3.9 or higher.

## Setup

1. Clone the repository
```bash
git clone https://github.com/FreeLike76/MeshPose
```

2. Create and activate python environment
```bash
conda create -n mpose python=3.9
conda activate mpose
```

3. Go to the library directory
```bash
cd MeshPose/meshpose
```

4. (Optional) Install pytorch
    
    __CUDA/CPU__
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    __CPU-only__
    ```bash
    pip install torch torchvision torchaudio
    ```
5. Install other dependencies
```bash
pip install -r requirements.txt
```

6. Install the library
```bash
pip install .
```

## Usage
Root directory provides 4 scripts for testing the library. All you need is a 3D Scanner App project to work with.