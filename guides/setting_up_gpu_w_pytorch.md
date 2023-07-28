* To be able to see the GPU(s) and usage, the nvtop package is needed.

        sudo apt install nvtop

* For WSL, nvtop installed using above approach might not work. Instead, build from source: https://github.com/Syllo/nvtop#nvtop-build

---

## Setting up NVIDIA CUDA on WSL (Windows Subsystem for Linux)
Follow the steps here: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

### Main steps
1. Install NVIDIA Driver for GPU support
  
    a. Select and download appropriate driver for GPU and Operating system https://www.nvidia.com/Download/index.aspx

2. Install WSL 2 (follow instruction in 2.2. Step 2 on https://docs.nvidia.com/cuda/wsl-user-guide/index.html )

3. Install CUDA Toolkit using WSL-Ubuntu Package. Follow command line instructions on https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local


---


## Install pytorch-gpu

1. If doing this for the first time (or perhaps just in general), preferably do this in a separate conda environment, so that if anything breaks during installation, this new conda environment can just be trashed without losing anything.

        * Create a new conda environment

                conda create --name <environment name>

        OR

        * If a .yml file is available,

                conda env create -f <filename>.yml

2. Obtain the appropriate (depending on the build (prefer stable), os, package (here conda), language (here python), and compute platform (CPU or Cuda versions for GPU)) command to run from https://pytorch.org/. Example:

        conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

3. Check that pytorch works (and with GPU) with the following series of commands in a WSL (Ubuntu) window/session, inside the conda pytorch environment

        python
        import torch

    a. To check if pytorch works

        a = torch.randn(5,3)
        print(a)

    b. To check if it works with GPU(s)

        torch.cuda.is_available()

        Output should be: "True"

