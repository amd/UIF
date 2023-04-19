# Unified Inference Frontend - PyTorch Tools

| Publisher | Built By | Multi-GPU Support |
| --------- | -------- | ----------------- |
| AMD       | AMD      | Yes               |

### Description 
Unified Inference Frontend (UIF) accelerates deep learning inference solutions on AMD compute platforms.  This container provides PyTorch implementation of UIF for AMD Instinct™ GPUs.  It includes tools, libraries, models and example designs optimized for the GPU.  For more information about UIF, including the use of this container, refer to https://github.com/amd/UIF


### Labels
* Deep Learning
* Frameworks
* Machine Learning
* Torch library


### Pull Command

```

    docker pull amdih/uif-pytorch:uif1.1_rocm5.4.1_vai3.0_py3.7_pytorch1.12
    
```

## Overview 

The UIF Pytorch Tools Docker container includes:

* ROCm™ 5.4.1
* ROCm™ Pytorch 1.12
* ROCm™ MIGraphX inference engine
* Scripts for downloading pretrained models from the Model Zoo
* Sample Application
* Instructions and tools for optimizing models using quantization and pruning

## Single-Node Server Requirements

| CPUs | GPUs | Operating Systems | ROCm™ Driver | Container Runtimes |
| -----| ------ | ----------------- | ------------ | ------------------ |
| X86_64 CPU(s) | AMD Instinct™ MI200 GPU(s) <br> AMD Instinct™ MI100 GPU(s) <br> Radeon Instinct™ MI50(S) | Ubuntu 20.04 <br> RedHat 8 | ROCm v5.x compatibility | [Docker Engine](https://docs.docker.com/engine/install/) <br> [Singularity](https://sylabs.io/docs/) |

## Running Containers - Using Docker
Use the following instructions to launch the Docker container for the application. 

```
    docker run -it --cap-add=SYS_PTRACE --security-opt 
    seccomp=unconfined --device=/dev/kfd 
    --device=/dev/dri --group-add video --ipc=host --shm-size 8G 	
    amdih/uif-pytorch:uif1.1_rocm5.4.1_vai3.0_py3.7_pytorch1.12
    
```

## Running Application

Follow the instructions at https://github.com/amd/UIF to set up models, run examples, deploy models or get tips on debugging and profiling.


## Licensing Information
________________________________________
Your use of this application is subject to the terms of the applicable component-level license identified below. To the extent any subcomponent in this container requires an offer for corresponding source code, AMD hereby makes such an offer for corresponding source code form, which will be made available upon request. By accessing and using this application, you are agreeing to fully comply with the terms of this license. If you do not agree to the terms of this license, do not access or use this application.

### Disclaimer
________________________________________
The information contained herein is for informational purposes only, and is subject to change without notice. In addition, any stated support is planned and is also subject to change. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.
 
### Notices and Attribution
________________________________________
© 2023 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD Arrow logo, Instinct, Radeon Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc.

Docker and the Docker logo are trademarks or registered trademarks of Docker, Inc. in the United States and/or other countries. Docker, Inc. and other parties may also have trademark rights in other terms used herein. Linux® is the registered trademark of Linus Torvalds  in the U.S. and other countries.     

All other trademarks and copyrights are property of their respective owners and are only mentioned for informative purposes.    
