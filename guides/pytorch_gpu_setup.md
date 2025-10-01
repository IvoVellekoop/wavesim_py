## [Setting up NVIDIA CUDA on WSL (Windows Subsystem for Linux)](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### Main steps

1. Install NVIDIA Driver for GPU support. Select and download
   appropriate [driver](https://www.nvidia.com/Download/index.aspx) for GPU and Operating system

2. Install WSL 2 (follow instruction in [2.2. Step 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html))

3. Install CUDA Toolkit using WSL-Ubuntu Package, following
   the [command line instructions](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local).

## Install pytorch-gpu

1. If doing this for the first time (or perhaps just in general), preferably do this in a separate conda environment, so
   that if anything breaks during installation, this new conda environment can just be trashed without losing anything.

    * Create a new conda environment
    ```
    conda create --name <environment name>
    ```

    * OR, if a .yml file is available,
    ```
    conda env create -f <filename>.yml
    ```

2. Obtain the appropriate (depending on the build (prefer stable), os, package (here conda), language (here, Python),
   and compute platform (CPU or CUDA versions for GPU)) command to run from https://pytorch.org/. Example:
    ```
    python -m pip install torch  --index-url https://download.pytorch.org/whl/cu126
    ```

3. Check that pytorch works (and with GPU) with the following series of commands in a WSL (Ubuntu) window/session,
   inside the conda pytorch environment
    ```
    python
    
    ```

    * To check if pytorch works
    ```
    a = torch.randn(5,3)
    print(a)
    ```

    * To check if it works with GPU(s)
    ```
    torch.cuda.is_available()
    ```
   Output should be: "True"

### Monitoring GPU usage

* To be able to see the GPU(s) and usage, the [nvtop](https://github.com/Syllo/nvtop) package is quite useful.
    ```
    sudo apt install nvtop
    ```

* For WSL, nvtop installed using above approach might not work.
  Instead, [build from source](https://github.com/Syllo/nvtop#nvtop-build)

* For Windows, nvtop is not available, but an alternate tool [nvitop](https://pypi.org/project/nvitop/) can be used
  instead.
