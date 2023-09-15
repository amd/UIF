<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.2 User Guide </h1>
    </td>
 </table>

# Unified Inference Frontend

Unified Inference Frontend (UIF) consolidates the following compute platforms under one AMD inference solution with unified tools and runtime:

- AMD EPYC&trade; and AMD Ryzen&trade; processors
- AMD Instinct&trade; and AMD Radeon&trade; GPUs
- AMD Versal&trade; Adaptive SoCs
- Field Programmable Gate Arrays (FPGAs)

UIF accelerates deep learning inference applications on all AMD compute platforms for popular machine learning frameworks, including TensorFlow, PyTorch, and ONNXRT. It consists of tools, libraries, models, and example designs optimized for AMD platforms. These enable deep learning application and framework developers to enhance inference performance across various workloads, including computer vision, natural language processing, and recommender systems. 

# Release Highlights

UIF 1.2 adds support for AMD Radeon&trade; GPUs in addition to AMD Instinct&trade; GPUs. Currently, [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX) is the acceleration library for both Radeon and Instinct GPUs for Deep Learning Inference. UIF supports 50 optimized models for Instinct and Radeon GPUs and 84 for EPYC CPUs. The AMD Vitis&trade; AI Optimizer tool is released as part of the Vitis AI 3.5 stack. UIF Quantizer is released in the PyTorch and TensorFlow Docker® images. Leveraging the UIF Optimizer and Quantizer enables performance benefits for customers when running with the MIGraphX and ZenDNN backends for Instinct and Radeon GPUs and EPYC CPUs, respectively. This release also adds MIGraphX backend for [AMD Inference Server](https://github.com/Xilinx/inference-server). This document provides information about downloading, building, and running the UIF v1.2 release.

The highlights of this release are as follows:

AMD Radeon&trade; GPU:
* Support for AMD Radeon&trade; PRO V620 and W6800 GPUs.
  For more information about the product, see https://www.amd.com/en/products/professional-graphics/amd-radeon-pro-w6800.
* Tools for optimizing inference models and deploying inference using the AMD ROCm™ platform. 
* Inclusion of the [rocAL](https://docs.amd.com/projects/rocAL/en/docs-5.5.0/user_guide/ch1.html) library.

Model Zoo:
* Expanded set of models for AMD CPUs and new models for AMD GPUs.

ZenDNN:
* TensorFlow, PyTorch, and ONNXRT with ZenDNN packages for download (from the ZenDNN web site)

ROCm:
* Docker containers containing tools for optimizing models for inference
* 50 models enabled to run on AMD ROCm platform using MIGraphX inference engine
* Up to 5.3x the throughput (images/second) running PT-OFA-ResNet50 with 78% pruned FP16 model on an AMD MI100 accelerator powered production server compared to the baseline FP32 PT- ResNet50v1.5 model. ([ZD-041](#zd041))
* Docker containers for running AMD Inference Server

AMD Inference Server provides a common interface for all inference modes:
  * Common C++ and server APIs for model deployment
  * Backend interface for using TensorFlow/PyTorch in inference for ZenDNN
  * Additional UIF 1.2 optimized models examples for Inference Server
  * Integration with KServe

[Introducing Once-For-All (OFA)](/docs/2_model_setup/uifmodelsetup.md#213-once-for-all-ofa-efficient-model-customization-for-various-platforms), a neural architecture search method that efficiently customizes sub-networks for diverse hardware platforms, avoiding high computation costs. OFA can achieve up to 1.69x speedup on MI100 GPUs compared to ResNet50 baselines.

# Prerequisites

The following prerequisites must be met for this release of UIF:
| Component          | Supported Hardware                                       |
|--------------------|---------------------------------------------------------|
| CPU                | AMD EPYC 9004 or 7003 Series Processors                |
| GPU                | AMD Radeon™ PRO V620 and W6800, AMD Instinct™ MI200 or MI100 Series GPU                |
| FPGA/AI Engine     | AMD Zynq™ SoCs or Versal devices supported in Vitis AI 3.5<br>**Note**: The inference server currently supports Vitis AI 3.0 devices|
                                     
| Component             | Supported Software                                    |
|-----------------------|-------------------------------------------------------|
| Operating Systems    | Ubuntu® 20.04 LTS and later, Red Hat® Enterprise Linux® 8.0 and later, CentOS 7.9 and later |
| ZenDNN                | Version 4.0 for AMD EPYC CPU                          |
| MIGraphX              | Version 2.6 for AMD Instinct GPU                      |
| Vitis AI              | Version 3.5 for FPGA/AIE, Model Zoo                   |
| Inference Server      | Version 0.4                                           |


## Getting Started with UIF v1.2

### Step 1: Installation 

The UIF software is made available through Docker Hub. The tools container contains the quantizer, compiler, and runtime for AMD Instinct GPUs and EPYC CPUs. The following page provides the instructions to install UIF:

- <a href="/docs/1_installation/installation.md#11-pull-pytorchtensorflow-docker-for-gpu-users">1.1: Pull PyTorch/TensorFlow Docker (for GPU Users)</a>
- <a href="/docs/1_installation/installation.md#12-pull-pytorchtensorflow-docker-for-fpga-users">1.2: Pull PyTorch/TensorFlow Docker (for FPGA Users)</a>
- <a href="/docs/1_installation/installation.md#13-install-zendnn-package-for-cpu-users">1.3: Install ZenDNN Package (for CPU Users)</a>
- <a href="/docs/1_installation/installation.md#14-get-the-inference-server-docker-image-for-model-serving">1.4: Get the Inference Server Docker Image (for Model Serving)</a>
 
### Step 2: Model Setup

The UIF Model Zoo includes optimized deep learning models to speed up the deployment of deep learning inference on AMD platforms. These models cover different applications, including but not limited to ADAS/AD, medical, video surveillance, robotics, and data center. Go to the following pages to learn how to download and set up the pre-compiled models for target platforms: 

 - <a href="/docs/2_model_setup/uifmodelsetup.md#21-uif-model-zoo-introduction">2.1: UIF Model Zoo Introduction</a>
 - <a href="/docs/2_model_setup/uifmodelsetup.md#22-get-zendnn-models-from-uif-model-zoo">2.2: Get ZenDNN Models from UIF Model Zoo</a>
 - <a href="/docs/2_model_setup/uifmodelsetup.md#23-get-migraphx-models-from-uif-model-zoo">2.3: Get MIGraphX Models from UIF Model Zoo</a>
 - <a href="/docs/2_model_setup/uifmodelsetup.md#24-set-up-migraphx-ymodel">2.4: Set Up MIGraphX YModel</a>
 - <a href="/docs/2_model_setup/uifmodelsetup.md#25-get-vitis-ai-models-from-uif-model-zoo">2.5: Get Vitis AI Models from UIF Model Zoo</a>
 - <a href="/docs/2_model_setup/gpu_model_example.md">2.6: GPU Model Example</a>
 
 ### Step 3: Run Examples

- <a href="/docs/3_run_example/runexample-script.md">3.1: Run a CPU Example</a>
- <a href="/docs/3_run_example/inference_server_example.md">3.2: Run an Example with the Inference Server</a>
- <a href="/docs/3_run_example/runexample-migraphx.md">3.3: Run an Example with MIGraphX</a>

### Step 4: Deploy Your Own Model

The following pages outline how to prune, quantize, and deploy the trained model on different target platforms to check performance optimization:

- <a href="/docs/4_deploy_your_own_model/prune_model/prunemodel.md">4.1: Prune Model with UIF Optimizer</a>
- <a href="/docs/4_deploy_your_own_model/quantize_model/quantizemodel.md">4.2: Quantize Model with UIF Quantizer for Target Platforms</a>
- <a href="/docs/4_deploy_your_own_model/deploy_model/deployingmodel.md">4.3: Deploy Model for Target Platforms</a>
- <a href="/docs/4_deploy_your_own_model/serve_model/servingmodelwithinferenceserver.md">4.4: Serve Model with Inference Server</a>

### Step 5: Debugging and Profiling

The following pages outline debugging and profiling strategies:

 - <a href="/docs/5_debugging_and_profiling/debugging_and_profiling.md#51-debug-on-gpu">5.1: Debug on GPU</a>
 - <a href="/docs/5_debugging_and_profiling/debugging_and_profiling.md#52-debug-on-cpu">5.2: Debug on CPU</a>
 - <a href="/docs/5_debugging_and_profiling/debugging_and_profiling.md#53-debug-on-fpga">5.3: Debug on FPGA</a>

 <hr/>

 [Next >](/docs/1_installation/installation.md)

 <hr/>


# License

UIF is licensed under [Apache License Version 2.0](LICENSE). Refer to the [LICENSE](LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.

# AMD Copyright Notice and Disclaimer

© 2022–2023 Advanced Micro Devices, Inc. All rights reserved

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.

THIS INFORMATION IS PROVIDED “AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

AMD, the AMD Arrow logo, ROCm™, ROCm™, Radeon™, AMD Instinct™, Radeon Instinct™, Radeon Pro™, RDNA™, CDNA™ and combinations thereof are trademarks of Advanced Micro Devices, Inc. 

The CentOS Marks are trademarks of Red Hat, Inc. Docker and the Docker logo are trademarks or registered trademarks of Docker, Inc. Kubernetes is a registered trademark of The Linux Foundation. Linux is the registered trademark of Linus Torvalds in the U.S. and other countries. Red Hat and the Shadowman logo are registered trademarks of Red Hat, Inc. www.redhat.com in the U.S. and other countries.
Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd.

Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. 

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

ROCm is made available by Advanced Micro Devices, Inc. under the open source license identified in the top-level directory for the library in the repository on Github.com (Portions of ROCm are licensed under MITx11 and UIL/NCSA. For more information on the license, review the license.txt in the top-level directory for the library on Github.com). The additional terms and conditions below apply to your use of ROCm technical documentation.

AQL PROFILER IS SUBJECT TO THE LICENSE AGREEMENT ENCLOSED IN THE DIRECTORY FOR THE AQL PROFILER AND IS AVAILABLE HERE: /usr/share/doc/hsa-amd-aqlprofile1.0.0/copyright. BY USING, INSTALLING, COPYING, OR DISTRIBUTING THE AQL PROFILER, YOU AGREE TO THE TERMS AND CONDITIONS OF THIS LICENSE AGREEMENT. IF YOU DO NOT AGREE TO THE TERMS OF THIS AGREEMENT, DO NOT INSTALL, COPY, OR USE THE AQL PROFILER.

AOCC CPU OPTIMIZATIONS BINARY IS SUBJECT TO THE LICENSE AGREEMENT ENCLOSED IN THE DIRECTORY FOR THE BINARY AND IS AVAILABLE HERE: /opt/rocm-5.0.0/share/doc/rocm-llvm-alt/EULA. BY USING, INSTALLING, COPYING, OR DISTRIBUTING THE AOCC CPU OPTIMIZATIONS, YOU AGREE TO THE TERMS AND CONDITIONS OF THIS LICENSE AGREEMENT. IF YOU DO NOT AGREE TO THE TERMS OF THIS AGREEMENT, DO NOT INSTALL, COPY, OR USE THE AOCC CPU OPTIMIZATIONS.

#### ZD036:

Testing conducted by AMD Performance Labs as of Thursday, January 12, 2023, on the ZenDNN v4.0 software library, Xilinx Vitis AI Model Zoo 3.5, on test systems comprising of AMD Eng Sample of the EPYC 9004 96-core processor, dual socket, with hyperthreading on, 2150 MHz CPU frequency (Max 3700 MHz), 786GB RAM (12 x 64GB DIMMs @ 4800 MT/s; DDR5 - 4800MHz 288-pin Low Profile ECC Registered RDIMM 2RX4), NPS1 mode, Ubuntu® 20.04.5 LTS version, kernel version 5.4.0-131-generic, BIOS TQZ1000F, GCC/G++ version 11.1.0, GNU ID 2.31, Python 3.8.15, AOCC version 4.0, AOCL BLIS version 4.0, TensorFlow version 2.10. Pruning was performed by the Xilinx Vitis AI pruning and quantization tool v3.5. Performance may vary based on use of latest drivers and other factors. ZD036

#### ZD041:

Testing conducted by AMD Performance Labs as of Wednesday, January 18, 2023, on test systems comprising of: AMD MI100, 1200 MHz CPU frequency, 8x32GB GPU Memory, NPS1 mode, Ubuntu® 20.04 version, kernel version 4.15.0-166-generic, BIOS 2.5.6, GCC/G++ version 9.4.0, GNU ID 2.34, Python 3.7.13, xcompiler version 3.5.0, pytorch-nndct version 3.5.0, xir version 3.5.0, target_factory version 3.5.0, unilog version 3.5.0, ROCm version 5.4.1.50401-84~20.04. Pruning was performed by the Xilinx Vitis AI pruning and quantization tool v3.5. Performance may vary based on use of latest drivers and other factors. ZD-041




