# Unified Inference Frontend
Unified Inference Frontend (UIF) is an effort to consolidate the following compute platforms under one AMD inference solution with unified tools and runtime:
- AMD EPYC&trade;
- AMD Ryzen&trade;
- AMD CDNA&trade;
- AMD RDNA&trade;
- Versal&reg; ACAP
- Field Programmable Gate Arrays (FPGAs)

UIF accelerates deep learning inference applications on all AMD compute platforms for popular machine learning frameworks, including TensorFlow, PyTorch, and ONNXRT. It consists of tools, libraries, models, and example designs optimized for AMD platforms that enable deep learning applications and framework developers to improve inference performance across various workloads such as computer vision, natural language processing, and recommender systems.

# Unified Inference Frontend 1.0
UIF 1.0 targets AMD EPYC&trade; CPUs. Currently, AMD EPYC&trade; CPUs use the ZenDNN library (for more information, see [Zen DNN library](https://github.com/amd/ZenDNN)) as the accelerator for deep learning inference. UIF 1.0 enables Vitis&trade; AI Model Zoo for TensorFlow+ZenDNN and PyTorch+ZenDNN. The Vitis AI Optimizer optimizes the models provided to developers as AMD-optimized models that show performance benefits when run on the ZenDNN backend. UIF 1.0 release also introduces TensorFlow+ZenDNN and PyTorch+ZenDNN backends for AMD Inference Server (for more information, see [AMD Inference Server](https://github.com/Xilinx/inference-server)). This document provides information about downloading, building, and running the UIF 1.0 release.

# Table of Contents

- [Release Highlights](#uif-10-release-highlights)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [ZenDNN Installation](#zendnn-installation)
  - [UIF Model Setup](#uif-model-setup)
  - [Inference Server Setup](#inference-server-setup)
- [Quick Start](#quick-start)
  - [Run Examples with TensorFlow+ZenDNN](#run-examples-with-tensorflowzendnn)
  - [Run Examples with PyTorch+ZenDNN](#run-examples-with-pytorchzendnn)
  - [Run Examples with Inference Server](#run-examples-with-inference-server)
- [Sample Run with TensorFlow+ZenDNN](#sample-run-with-tensorflowzendnn)
- [License](#license)
- [Technical Support](#technical-support)
- [Disclaimer](#Disclaimer)

# UIF 1.0 Release Highlights
UIF 1.0 highlights include:
* TensorFlow/PyTorch installer packages for ZenDNN
* 50+ float, int8 quantized, and pruned models enabled to run on TensorFlow+ZenDNN, PyTorch+ZenDNN
* The following standard interface for all inference modes supported by the inference server:
  * Common C++ and Server APIs for model deployment
  * Backend interface for using TensorFlow/PyTorch+ZenDNN for inference

# Prerequisites
UIF 1.0 prerequisites include:
* Hardware  
  * CPU: AMD EPYC&trade; [7002](https://www.amd.com/en/processors/epyc-7002-series) and [7003](https://www.amd.com/en/processors/epyc-7003-series) Series Processors
* Software
  * OS: Ubuntu® 18.04 LTS and later, Red Hat® Enterprise Linux® (RHEL) 8.0 and later, and CentOS 7.9 and later
  * ZenDNN 3.3
  * Vitis AI 2.5 Model Zoo
  * Inference Server 0.3.0

# Installation

## ZenDNN Installation
Perform the following steps to install TensorFlow and PyTorch built with ZenDNN:
### TensorFlow+ZenDNN
To run inference on the TensorFlow model using ZenDNN, you must first download and install the TensorFlow+ZenDNN package. Perform the following steps to complete the TensorFlow+ZenDNN installation:
1. Download the TensorFlow+ZenDNN v3.3 release package from [AMD Developer Central](https://developer.amd.com/zendnn/).

2. Unzip the package. For example: TF_v2.9_ZenDNN_v3.3_Python_v3.8.zip.
    ```
   unzip TF_v2.9_ZenDNN_v3.3_Python_v3.8.zip
    ```
3. Ensure that you have the conda environment installed, and execute the following commands:

    ```
    cd TF_v2.9_ZenDNN_v3.3_Python_v*/
    source scripts/TF_ZenDNN_setup_release.sh
    ```
TensorFlow+ZenDNN installation completes.

### PyTorch+ZenDNN
To run inference on the PyTorch model using ZenDNN, you must first download and install the PyTorch+ZenDNN package. Perform the following steps to complete the PyTorch+ZenDNN installation:
1. Download PTv1.11+ZenDNNv3.3 release package from [AMD Developer Central](https://developer.amd.com/zendnn/).

2. Unzip the package. For example: PT_v1.11.0_ZenDNN_v3.3_Python_v3.8.zip.
    ```
    unzip PT_v1.11.0_ZenDNN_v3.3_Python_v3.8.zip
    ```
3. Ensure that you have the conda environment installed, and execute the following commands:

    ```
    cd PT_v1.11.0_ZenDNN_v3.3_Python_v*/ZenDNN/
    source scripts/PT_ZenDNN_setup_release.sh
    ```
    PyTorch+ZenDNN installation completes.

## UIF Model Setup
You can access the models supported by UIF 1.0 from [UIF Developer Site](https://www.xilinx.com/member/uif_developer.html).

## Inference Server Setup

AMD Inference Server is integrated with [ZenDNN](https://developer.amd.com/zendnn/) optimized libraries. For more information on the AMD Inference Server, see the [GitHub page](https://github.com/Xilinx/inference-server) or the [documentation](https://xilinx.github.io/inference-server/main/index.html).

**Note:** To use AMD Inference Server, you need Git and Docker installed on your machine.

1. Clone the Inference Server.

    1. Clone the inference-server repo
       ```
       git clone https://github.com/Xilinx/inference-server
       ```
    2. Navigate to the `inference-server`
       ```
       cd inference-server
       ```

2. Download the C++ package for TensorFlow/PyTorch+ZenDNN.

    1. Go to https://developer.amd.com/zendnn/
    2. Download the file
        1. For TensorFlow: TF_v2.9_ZenDNN_v3.3_C++_API.zip
        2. For PyTorch: PT_v1.11.0_ZenDNN_v3.3_C++_API.zip
    3. Copy the downloaded package within the repository. Use the package for the next steps in the setup.

3. Build the Docker with TensorFlow/PyTorch+ZenDNN.

    Currently, these images are not pre-built, so you must build them. You must enable Docker BuildKit by setting `export DOCKER_BUILDKIT=1` in the environment.  To build the Docker image with TensorFlow/PyTorch+ZenDNN, use the following command:
    1. For TensorFlow
        ```
        ./proteus dockerize --no-vitis --tfzendnn=TF_v2.9_ZenDNN_v3.3_C++_API.zip
        ```
    2. For PyTorch
        ```
        ./proteus dockerize --no-vitis --ptzendnn=PT_v1.11.0_ZenDNN_v3.3_C++_API.zip
        ```
    `--no-vitis` flag is provided to build the Docker without Vitis components.

    This builds a Docker image with all the dependencies required for the AMD Inference Server and sets up TensorFlow/PyTorch+ZenDNN within the image for further usage.

    **Note:** The downloaded package must be inside the inference-server folder since the Docker will not be able to access the file outside the repository.

# Quick Start
The quick start section introduces using the ZenDNN optimized models with TensorFlow and PyTorch and getting started using the AMD Inference Server.

### Run Examples with TensorFlow+ZenDNN

Install TensorFlow+ZenDNN. For more information, see the [Installation section](#tensorflowzendnn).

This tutorial uses ResNet50 as an example. Download the ResNet50 model. For more information, see the [UIF Model Setup section](#uif-model-setup).

1. Unzip the model package.
    ```
    unzip tf_resnetv1_50_imagenet_224_224_6.97G_2.5_1.0_Z3.3.zip
    ```

2. Run the run_bench.sh script for FP32 model and run_bench_quant.sh for Quantized model to benchmark the performance of ResNet50

    ```
    cd tf_resnetv1_50_imagenet_224_224_6.97G_2.5_1.0_Z3.3
    bash run_bench.sh 64 640
    bash run_bench_quant.sh 64 640
    ```
Similarly, use the `run_eval` scripts for validating the accuracy. To set up the validation data, refer to the readme files provided with the model package.

### Run Examples with PyTorch+ZenDNN

Install PyTorch+ZenDNN. For more information, see the [Installation section](#pytorchzendnn).

This tutorial uses personreid-resnet50 as an example. Download the personreid-resnet50 model as described in the [UIF Model Setup section](#uif-model-setup).

1. Unzip the model package.
    ```
    unzip pt_personreid-res50_market1501_256_128_5.3G_2.5_1.0_Z3.3.zip
    ```

2. Check the <code>readme.md</code> file for required dependencies. Run the `run_bench.sh` script for FP32 model to benchmark the performance of personreid-resnet50.

    ```
    cd pt_personreid-res50_market1501_256_128_5.3G_2.5_1.0_Z3.3
    bash run_bench.sh 64 640
    ```
Similarly, use the `run_eval` scripts for validating the accuracy. To set up the validation data, refer to the readme files provided with the model package.
## Run Examples with Inference Server

There are two examples provided in the repo (Python API and C++ API) for both TensorFlow and PyTorch.

### Get Objects (Models/Images)
Run the following command to get some git lfs assets for examples.

```
git lfs fetch --all
git lfs pull
```

To run the examples and test cases, download some models as follows:

1. TensorFlow+ZenDNN

    Run the following command to download a ResNet50 TensorFlow model.
    ```
    ./proteus get --tfzendnn
    ```
2.  PyTorch+ZenDNN

    Run the command below to download a ResNet50 PyTorch model.
    ```
    ./proteus get --ptzendnn
    ```

### Set Up Docker Container

1. Run the container.

    By default, the stable dev Docker image is built from [Inference Server Setup](#inference-server-setup). To run the container, use the following command:
    ```
    ./proteus run --dev
    ```

2. Build the AMD Inference Server.

    Now that the environment is set up within the Docker container, you can build the AMD Inference Server. The following command builds the stable debug build of the AMD Inference Server:
    ```
    ./proteus build --debug
    ```
    **Note:** If you are switching containers and the build folder already exists in the inference-server folder, use `--regen --clean` flags to regenerate CMakeFiles and do a clean build to avoid any issues.

3. For PyTorch+ZenDNN only:

    Convert the downloaded PyTorch eager model to TorchScript Model ([Exporting to TorchScript docs](https://pytorch.org/tutorials/advanced/cpp_export.html#converting-to-torch-script-via-tracing)).

    To convert the model to the TorchScript model, follow these steps:

    1. Use the PyTorch Python API. Install requirements with the following command:
        ```
        pip3 install -r tools/zendnn/requirements.txt
        ```
    2. To convert the model to the TorchScript Model, execute the following:
        ```
        python tools/zendnn/convert_to_torchscript.py --graph external/pytorch_models/resnet50_pretrained.pth
        ```

#### Python API

Both the Python examples do the following:
1. Start the AMD Inference Server on HTTP port 8998.
2. Load the AMD Inference Server with the specified model file.
3. Read the specified image and perform preprocessing.
4. Send the images to the AMD Inference Server over HTTP, wrapped in a specific format.
5. Get the result back from the AMD Inference Server.
6. Postprocess, if required, and display the output.

Follow the steps below to run the examples:
1. TensorFlow+ZenDNN

    ```
    python examples/python/tf_zendnn.py --graph ./external/tensorflow_models/resnet_v1_50_baseline_6.96B.pb --image_location ./tests/assets/dog-3619020_640.jpg
    ```
2. PyTorch+ZenDNN

    ```
    python examples/python/pt_zendnn.py --graph ./external/pytorch_models/resnet50_pretrained.pt --image_location ./tests/assets/dog-3619020_640.jpg
    ```


#### C++ API

The C++ API bypasses the HTTP server and connects directly to the Inference Server. The flow is as follows:
1. Load the AMD Inference Server for the specified model file.
2. Read the specified image and perform preprocessing.
3. Pack the data into an interface object and push it to a queue.
4. Retrieve the result back from the AMD Inference Server.
5. Postprocess, if required, and display the output.

The C++ example is built when the server is built according to the available package. To build and run the example, use the following command.

1. TensorFlow+ZenDNN

    ```
    ./proteus build --debug && ./build/Debug/examples/cpp/tf_zendnn_client
    ```
2. PyTorch+ZenDNN

    ```
    ./proteus build --debug && ./build/Debug/examples/cpp/pt_zendnn_client
    ```
# Sample Run with TensorFlow+ZenDNN

The repo consists of a sample to run a demo application. The demo loads images from a given folder, infers on the images with the given TensorFlow model, and prints out Top1 and Top5 classes.

## Preparation

1. Set up the environment.

    Install TensorFlow+ZenDNN. For more information, see the [Installation section](#tensorflowzendnn). Pillow is required, and you can install it with the following command.
    ```
    pip install pillow
    ```

2. Download the models.

    Download the TensorFlow Image Recognition (ResNet50, InceptionV3, MobileNetv1, VGG16) model from the model zoo. For more information, see the [model setup section](#uif-model-setup).

3. Get data.

    Due to copyright reasons, images are not included with this sample.

    To use the sample, create a directory and copy the required images into the folder. The images in this folder are used for inference.

## Run Sample

To run the sample, change the directory into the samples directory. Use the following command with the correct parameters:
```
python tf_sample.py \
    --model_file <model_file_path> \
    --input_height <input_height> \
    --input_width <input_width> \
    --input_layer <input_layer_name> \
    --output_layer <output_layer_name> \
    --data_resize_method <preprocess_method> \
    --batch_size <batch_size> \
    --data_location <data_location_path>
```

### Parameter Descriptions
```
--model_file:         Graph/model to be used for inference (.pb file)
--input_height:       Height for the image
--input_width:        Width for the image
--input_layer:        Name of the input node of the model
--output_layer:       Name of the output node of the model
--data_resize_method: Preprocessing method to be used for the model (ResNet50, VGG16: cropwithmeansub. InceptionV3, MobileNetv1: bilinear)
--batch_size:         Batch size to be used for inference. If the total number of images is less than the batch size given, the number of images will be used as the batch size
--data_location:      Path to the directory containing the images to be used for inference
```

### Example
This tutorial uses ResNet50 as an example. To run the model, refer to [TensorFlow+ZenDNN Example Run Section](#run-examples-with-tensorflowzendnn). To run for a single image:
```
python tf_sample.py \
 --model_file /path/to/resnet_v1_50_inference.pb \
 --input_height 224 \
 --input_width 224 \
 --input_layer input \
 --output_layer resnet_v1_50/predictions/Reshape_1 \
 --data_resize_method cropwithmeansub \
 --batch_size 64 \
 --data_location /path/to/image_directory
```
# License

UIF is licensed under [Apache License Version 2.0](LICENSE). Refer to the [LICENSE](LICENSE) file for the full license text and copyright notice.

# Technical Support
Please email uif_support@amd.com for questions, issues, and feedback on UIF.

Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/amd/UIF/issues) page.

# Disclaimer
The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. Neither AMD nor any of its affiliates and subsidiaries (individually and collectively, “AMD”) assume any obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED "AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

AMD, the AMD Arrow logo, AMD EPYC&trade;, AMD Ryzen&trade;, AMD CDNA&trade;, AMD RDNA&trade;,  Versal&reg; ACAP, Xilinx, the Xilinx logo, Alveo&trade;, Artix&trade;, ISE&reg;, Kintex&trade;, Spartan&trade;, Virtex&trade;, Zynq&trade;, and other designated brands included herein and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd. Red Hat and the Shadowman logo are registered trademarks of Red Hat, Inc. www.redhat.com in the U.S. and other countries. The CentOS Marks are trademarks of Red Hat, Inc.


© 2022 Advanced Micro Devices, Inc. All rights reserved
