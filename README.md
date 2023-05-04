<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.1 User Guide </h1>
    </td>
 </table>

# Unified Inference Frontend

Unified Inference Frontend (UIF) is an effort to consolidate the following compute platforms under one AMD inference solution with unified tools and runtime:

- AMD EPYC&trade; processors
- AMD Instinct™ GPUs
- AMD Ryzen&trade; processors
- Versal&trade; ACAP
- Field Programmable Gate Arrays (FPGAs)

UIF accelerates deep learning inference applications on all AMD compute platforms for popular machine learning frameworks, including TensorFlow, PyTorch, and ONNXRT. It consists of tools, libraries, models, and example designs optimized for AMD platforms that enable deep learning applications and framework developers to improve inference performance across various workloads such as computer vision, natural language processing, and recommender systems.


![](/images/slide24.png)

* **Note:** WinML is supported on Windows OS only.

# Unified Inference Frontend 1.1

UIF 1.1 extends the support to AMD Instinct GPUs in addition to EPYC CPUs starting from UIF 1.0. Currently, [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX) is the acceleration library for Instinct GPUs for Deep Learning Inference. UIF 1.1 provides 45 optimized models for Instinct GPUs and 84 for EPYC CPUs. The Vitis&trade; AI Optimizer tool is released as part of the Vitis AI 3.0 stack. UIF Quantizer is released in the PyTorch and TensorFlow Docker® images. Leveraging the UIF Optimizer and Quantizer enables performance benefits for customers when running with the MIGraphX and ZenDNN backends for Instinct GPUs and EPYC CPUs, respectively. This release also adds MIGraphX backend for [AMD Inference Server](https://github.com/Xilinx/inference-server). This document provides information about downloading, building, and running the UIF 1.1 release.

## AMD Instinct GPU

UIF 1.1 targets support for AMD GPUs. While UIF 1.0 enabled Vitis AI Model Zoo for TensorFlow+ZenDNN and PyTorch+ZenDNN, UIF v1.1 adds support for AMD Instinct&trade; GPUs. 

UIF 1.1 also introduces tools for optimizing inference models. GPU support includes the ability to use AMD GPUs for optimizing inference as well the ability to deploy inference using the AMD ROCm™ platform. Additionally, UIF 1.1 has expanded the set of models available for AMD CPUs and introduces new models for AMD GPUs as well.

# Release Highlights

The highlights of this release are as follows:

ZenDNN:
* TensorFlow, PyTorch, and ONNXRT with ZenDNN packages for download (from the ZenDNN web site)
* 84 model packages containing FP32/BF16/INT8 models enabled to be run on TensorFlow+ZenDNN, PyTorch+ZenDNN and ONNXRT+ZenDNN
* Up to 20.5x the throughput (images/second) running Medical EDD RefineDet with the Xilinx Vitis AI Model Zoo 3.0 88% pruned INT8 model on 2P AMD Eng Sample: 100-000000894-04
of the EPYC 9004 96-core processor powered server with ZenDNN v4.0 compared to the baseline FP32 Medical EDD RefineDet model from the same Model Zoo. ([ZD-036](#zd036))
* Docker containers for running AMD Inference Server

ROCm:
* Docker containers containing tools for optimizing models for inference
* 30 quantized models enabled to run on AMD ROCm platform using MIGraphX inference engine
* Up to 5.3x the throughput (images/second) running PT-OFA-ResNet50 with the Xilinx Vitis AI Model Zoo 3.0 88% pruned FP16 model on an AMD MI100 accelerator powered production server compared to the baseline FP32 PT- ResNet50v1.5 model from the same Model Zoo. ([ZD-041](#zd041))
* Docker containers for running AMD Inference Server

AMD Inference Server provides a common interface for all inference modes:
  * Common C++ and server APIs for model deployment
  * Backend interface for using TensorFlow/PyTorch in inference for ZenDNN
  * Additional UIF 1.1 optimized models examples for Inference Server
  * Integration with KServe

# Prerequisites

The following prerequisites must be met for this release of UIF:

* Hardware based on target platform:
  * CPU: AMD EPYC [9004](https://www.amd.com/en/processors/epyc-9004-series) or [7003](https://www.amd.com/en/processors/epyc-7003-series) Series Processors
  * GPU: AMD Instinct&trade; [MI200](https://www.amd.com/en/graphics/instinct-server-accelerators) or [MI100](https://www.amd.com/en/products/server-accelerators/instinct-mi100) Series GPU
  * FPGA/AI Engine: Zynq&trade; SoCs or Versal devices supported in [Vitis AI 3.0](https://github.com/Xilinx/Vitis-AI)

* Software based on target platform:
  * OS: Ubuntu® 18.04 LTS and later, Red Hat® Enterprise Linux® (RHEL) 8.0 and later, CentOS 7.9 and later
  * ZenDNN 4.0 for AMD EPYC CPU
  * MIGraphX 2.4 for AMD Instinct GPU
  * Vitis AI 3.0 FPGA/AIE
  * Vitis AI 3.0 Model Zoo
  * Inference Server 0.3

## Implementing UIF 1.1

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
 
 
 ### Step 6: Deploying on PyTorch and Tensorflow

The following pages outline deploying strategies on PyTorch and Tensorflow:

 - <a href="/docs/6_deployment_guide/PyTorch.md">PyTorch</a>
 - <a href="/docs/6_deployment_guide/Tensorflow.md">Tensorflow</a>

<hr/>

 [Next >](/docs/1_installation/installation.md)

 <hr/>


# License

UIF is licensed under [Apache License Version 2.0](LICENSE). Refer to the [LICENSE](LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://https://github.com/amd/UIF/issues) page.

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

Testing conducted by AMD Performance Labs as of Thursday, January 12, 2023, on the ZenDNN v4.0 software library, Xilinx Vitis AI Model Zoo 3.0, on test systems comprising of AMD Eng Sample of the EPYC 9004 96-core processor, dual socket, with hyperthreading on, 2150 MHz CPU frequency (Max 3700 MHz), 786GB RAM (12 x 64GB DIMMs @ 4800 MT/s; DDR5 - 4800MHz 288-pin Low Profile ECC Registered RDIMM 2RX4), NPS1 mode, Ubuntu® 20.04.5 LTS version, kernel version 5.4.0-131-generic, BIOS TQZ1000F, GCC/G++ version 11.1.0, GNU ID 2.31, Python 3.8.15, AOCC version 4.0, AOCL BLIS version 4.0, TensorFlow version 2.10. Pruning was performed by the Xilinx Vitis AI pruning and quantization tool v3.0. Performance may vary based on use of latest drivers and other factors. ZD036

#### ZD041:

Testing conducted by AMD Performance Labs as of Wednesday, January 18, 2023, on test systems comprising of: AMD MI100, 1200 MHz CPU frequency, 8x32GB GPU Memory, NPS1 mode, Ubuntu® 20.04 version, kernel version 4.15.0-166-generic, BIOS 2.5.6, GCC/G++ version 9.4.0, GNU ID 2.34, Python 3.7.13, xcompiler version 3.0.0, pytorch-nndct version 3.0.0, xir version 3.0.0, target_factory version 3.0.0, unilog version 3.0.0, ROCm version 5.4.1.50401-84~20.04. Pruning was performed by the Xilinx Vitis AI pruning and quantization tool v3.0. Performance may vary based on use of latest drivers and other factors. ZD-041




