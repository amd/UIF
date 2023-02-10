<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.1 User Guide </h1>
    </td>
 </tr>
 <tr>
 <td align="center"><h1>4.3: Deploy Model for Target Platforms</h1>
 </td>
 </tr>
</table>


# Table of Contents

- [4.3.1: Deploy Model for FPGA](#431-deploy-model-for-fpga)
  - [4.3.1.1: In-Framework](#4311-in-framework)
  - [4.3.1.2: Native](#4312-native)
- [4.3.2: Deploy Model for CPU](#432-deploy-model-for-cpu)
  - [4.3.2.1: Run UIF Models with ZenDNN](#4321-run-uif-models-with-zendnn)
  - [4.3.2.2: Run Custom Models with ZenDNN](#4322-run-custom-models-with-zendnn)
- [4.3.3: Deploy Model for GPU](#433-deploy-model-for-gpu)
  - [4.3.3.1: Preliminary Steps](#4331-preliminary-steps)
  - [4.3.3.2: Prepare the Example](#4332-prepare-the-example)
  - [4.3.3.3: Run the Example from Python](#4333-run-the-example-from-python)
  - [4.3.3.4: Set Up the Video and Capture the Model](#4334-set-up-the-video-and-capture-the-model)
  - [4.3.3.5: Add Code for Preprocessing Video Frames](#4335-add-code-for-preprocessing-video-frames)
  - [4.3.3.6: Run the Complete Look Over Video](#4336-run-the-complete-look-over-video)

  _Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._


# 4.3.1: Deploy Model for FPGA

## 4.3.1.1: In-Framework

WeGO (<u>W</u>hol<u>e</u> <u>G</u>raph <u>O</u>ptimizer) offers a smooth solution to deploy models on cloud DPU by integrating the Vitis&trade; AI Development kit with TensorFlow 1.x, TensorFlow 2.x, and PyTorch frameworks.

The following platforms are supported for WeGo:
* Versal™ AI Core series VCK5000-PROD

For more information on setting up the host and running WeGO examples, see the [WeGo section](https://github.com/Xilinx/Vitis-AI/tree/master/examples/wego).

## 4.3.1.2: Native

The following platforms are supported for UIF 1.1:

* Zynq® UltraScale+™ MPSoC ZU9EG, ZCU102
* Zynq UltraScale+ MPSoC ZU7EV, ZCU104
* Zynq UltraScale+ MPSoC, Kria KV260
* Versal AI Core series VC1902, VCK190, VCK5000-Prod
* Versal Edge AI Core series VE2082, VEK280  

### Run Models on Edge Platform

1. Refer to [MPSoC](https://github.com/Xilinx/Vitis-AI/tree/master/board_setup/mpsoc) and [Versal](https://github.com/Xilinx/Vitis-AI/tree/master/board_setup/vck190) to set up the board respectively.

2. Download the [VART runtime](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-3.0.0.tar.gz) and install it.
   ```
   tar -xzvf vitis-ai-runtime-3.0.0.tar.gz
   cd vitis-ai-runtime-3.0.0/2022.2/aarch64/centos
   rpm -ivh --force *.rpm
   ```
3. Download the pre-compiled model from [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo/model-list).

   Take `resnet_v1_101_tf` as an example. Copy the model to the board.

   ```
   tar -xzvf resnet_v1_101_tf-zcu102_zcu104_kv260-r3.0.0.tar.gz -C /usr/share/vitis_ai_library/models
   ```
4. Download the test examples from [Vitis AI library examples](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_library/samples). For `resnet_v1_101_tf` model, the `classification` example is used to test.

5. Cross-compile the `classification` example on the host, then copy the executable program to the target.

   ```
   cd classification
   bash build.sh
   ```
6. Run the program on the target:

   ```
   ./test_jpeg_classification resnet_v1_101_tf sample_classification.jpg
   ```
   To test the performance of the model, run the following command:
   ```
   ./test_performance_facedetect resnet_v1_101_tf test_performance_facedetect.list -t 8 -s 60

   -t: <num_of_threads>
   -s: <num_of_seconds>
   ```

### Run Models on Cloud Platform

1. Download `Vitis-AI`, enter the `Vitis-AI` directory, and then start the Docker® software. For more information, see the [Getting Started](https://github.com/Xilinx/Vitis-AI#getting-started) section in the Vitis AI™ development environment documentation.

2. For the `VCK5000-PROD` Versal card, follow the instructions in [Set Up the VCK5000 Accelerator Card](https://github.com/Xilinx/Vitis-AI/blob/master/board_setup/vck5000/board_setup_vck5000.rst) to set up the host.
    
3. Run Vitis AI Library examples on `VCK5000-PROD`. For more information, see [Run Vitis AI Library Samples](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_library/README.md#running-vitis-ai-library-examples-on-vck5000) in the Vitis AI documentation.

# 4.3.2: Deploy Model for CPU

## 4.3.2.1: Run UIF Models with ZenDNN

This section introduces using the ZenDNN optimized models with TensorFlow, PyTorch and ONNXRT.

### Run Examples with TensorFlow+ZenDNN

Install TensorFlow+ZenDNN. For more information, see the [Installation](/docs/1_installation/installation.md#131-tensorflowzendnn) section.

This tutorial uses ResNet50 as an example. Download the ResNet50 model. For more information, see the [UIF Model Setup section](/docs/2_model_setup/uifmodelsetup.md).

1. Unzip the model package:

    ```
    unzip tf_resnetv1_50_imagenet_224_224_6.97G_1.1_Z4.0.zip
    ```

2. Check the <code>readme.md</code> file for required dependencies. Run the `run_bench.sh` script for FP32 model and `run_bench_quant.sh` for the quantized model to benchmark the performance of ResNet-50:

    ```
    cd tf_resnetv1_50_imagenet_224_224_6.97G_1.1_Z4.0
    bash run_bench.sh 64 640
    bash run_bench_quant.sh 64 640
    ```

Similarly, use the `run_eval` scripts for validating the accuracy. To set up the validation data, refer to the readme files provided with the model package.

### Run Examples with PyTorch+ZenDNN

Install PyTorch+ZenDNN. For more information, see the [Installation](/docs/1_installation/installation.md#132-pytorchzendnn) section.

This tutorial uses personreid-resnet50 as an example. Download the personreid-resnet50 model as described in the [UIF Model Setup section](/docs/2_model_setup/uifmodelsetup.md).

1. Unzip the model package.

    ```
    unzip pt_personreid-res50_market1501_256_128_5.3G_1.1_Z4.0.zip
    ```

2. Check the <code>readme.md</code> file for required dependencies. Run the `run_bench.sh` script for FP32 model and `run_bench_quant.sh` for the quantized model to benchmark the performance of personreid-resnet50.

    ```
    cd pt_personreid-res50_market1501_256_128_5.3G_1.1_Z4.0
    bash run_bench.sh 64 640
    bash run_bench_quant.sh 64 640
    ```
Similarly, use the `run_eval` scripts for validating the accuracy. To set up the validation data, refer to the readme files provided with the model package.

### Run Examples with ONNXRT+ZenDNN

Install ONNXRT+ZenDNN. For more information, see the [Installation](/docs/1_installation/installation.md#133-onnxrtzendnn) section.

This tutorial uses ResNet50 as an example. Download the ResNet50 model as described in the [UIF Model Setup section](/docs/2_model_setup/uifmodelsetup.md).

1. Unzip the model package.

    ```
    unzip onnx_resnetv1_50_imagenet_224_224_6.97G_1.1_Z4.0.zip
    ```

2. Check the <code>readme.md</code> file for required dependencies. Run the `run_bench.sh` script for FP32 model and `run_bench_quant.sh` for the quantized model to benchmark the performance of ResNet50.

    ```
    cd onnx_resnetv1_50_imagenet_224_224_6.97G_1.1_Z4.0
    bash run_bench.sh 64 640
    bash run_bench_quant.sh 64 640
    ```
Similarly, use the `run_eval` scripts for validating the accuracy. To set up the validation data, refer to the readme files provided with the model package.

## 4.3.2.2: Run Custom Models with ZenDNN

### Float Models

   To run any single-precision (float) custom model on ZenDNN, follow the steps given in the [ZenDNN Installation section](/docs/1_installation/installation.md#13-install-zendnn-package-for-cpu-users) to install TensorFlow+ZenDNN, PyTorch+ZenDNN or ONNXRT+ZenDNN. Once installation is complete, the model can be run with standard inference steps. One such example is provided in the [example section](/docs/3_run_example/runexample-script.md#311-sample-run-with-tensorflowzendnn).

### Model Compression Techniques for ZenDNN

#### 1. Pruning a Deep Learning Model

To use the neural compression technique of [pruning a deep learning model](/docs/4_deploy_your_own_model/prune_model/prunemodel.md#411-pruning), follow the steps given in section [4.1: Prune Model with UIF Optimizer](/docs/4_deploy_your_own_model/prune_model/prunemodel.md). Once the pruned models are generated, they can be run on frameworks built with ZenDNN.

#### 2. Quantizing a Deep Learning Model

 Supporting quantization for AMD CPUs is done in two steps:

 1. Use the [UIF Quantizer tool](/docs/4_deploy_your_own_model/quantize_model/quantizemodel.md#422-quantize-tensorflow-models) to quantize a model.
 2. Run the quantized model generated in step 1 through the ZenDNN model converter tool to create ZenDNN optimized model which can be run on ZenDNN.

To make use of the ZenDNN model converter tool:

1. Set up the environment:
   1. Install conda.
   2. Set up the TensorFlow+ZenDNN environment by following the steps in the [ZenDNN Installation section](/docs/1_installation/installation.md#13-install-zendnn-package-for-cpu-users).
   3. Install up the model converter tool:

      From the `.whl` file provided for the model converter at `/tools/zendnn`, install using the following command:
      ```
      python -m pip install ModelConverter-0.1-py3-none-linux_x86_64.whl
      ```
2. Convert the quantized model to a ZenDNN optimized model:

   The quantized model which is generated with the UIF Quantizer tool for TensorFlow is given as input to the Model Converter tool.

   Run the model converter using the following command:
   ```
   model_converter --model_file <path/to/the/model> --out_location <path/to/output/directory>
   ```

   <B>Parameter Descriptions</B>
   ```
   --model_file      : Graph/model to be used for optimization.
   --out_location    : Path to where the optimized model should be saved.
   ```

   Example usage is as follows:

   ```
   model_converter \
   --model_file ~/quantized/quantized_pruned_19.56B.pb \
   --out_location ./outputs/
   ```
   
   The result is an optimized graph that will be saved at the desired output location. The model will be saved with the same name appended with `_amd_opt.pb`. In the example, the model will be saved as `quantized_pruned_19.56B_amd_opt.pb` to the `outputs` folder. This optimized model can then be run on AMD CPUs through ZenDNN. Refer to the [AMD page for ZenDNN](https://www.amd.com/en/developer/zendnn.html) for more info.

   **Note:** Currently only TensorFlow models quantized using the UIF Quantizer tool are supported with model converter tool.

   This model converter is tested to work with Resnetv1 models (ResNet50, ResNet101, ResNet152), Inception models (InceptionV1, InceptionV3, InceptionV4), VGG models (VGG16, VGG19), EfficientNet models (EfficientNet-S, EfficientNet-M, EfficientNet-L), and RefineDet variants.


# 4.3.3: Deploy Model for GPU

**Note:** This GPU example assumes you run inside a Docker image started as described in section [1.1.3: Pull a UIF Docker Image](/docs/1_installation/installation.md#113-pull-a-uif-docker-image) and have downloaded the Resnet50v1.5 model as described in section [2.3: Get MIGraphX Models from UIF Model Zoo](/docs/2_model_setup/uifmodelsetup.md#23-get-migraphx-models-from-uif-model-zoo).

The following example describes the steps needed to run GPU inference using MIGraphX using a model named `resnet50_fp32.onnx` from the Model Zoo.

For additional information and examples on running MIGraphX, refer to the [ROCm Deep Learning Guide.](https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4.1/page/Introduction_to_Deep_Learning_Guide.html) 

# 4.3.3.1: Preliminary Steps

1. Download and run a GPU Docker. For more information, refer to the installation instructions in [Installation](/docs/1_installation/installation.md).

    ```

      prompt% docker pull amdih/uif-pytorch:uif1.1_rocm5.4.1_vai3.0_py3.7_pytorch1.12 

      prompt% docker run -it –cap-add=SYS_PTRACE –security-opt seccomp=undefined --device=/dev/kfd --device=dri --group-add render --ipc=host --shm-size 8G amdih/uif-pytorch:uif1.1_rocm5.4.1_vai3.0_py3.7_pytorch1.12 - base 
    ```

2. Download a trained model for the GPU. For more information, refer to [Model Setup](/docs/2_model_setup/uifmodelsetup.md).

 
    ```

        prompt% cd ~ 

        prompt% git clone https://github.com/AMD/uif.git 

        prompt% cd uif/docs/2_model_setup

        prompt% python3 downloader.py 

    
    ```


    The following prompt appears: 

    ```

        input:pt 
        
        choose model 

        0 : all 

        1 : pt_resnet50v1.5_imagenet_224_224_8.2G_1.1_M2.4

        … 

        input num: 1

    ```  


    The ResNet50 v1.5 PyTorch model is selected.

    ```

    3. Choose model type.

    0: all 

    1: GPU 

    2: MI100 

    3: MI210

    ...

    ```
 

3. Select and download an MI-210 YModel:

    ```
    input num:3 
      pt_resnet50v1.5_imagenet_224_224_8.2G_1.1_M2.4_MI210.zip  
                                              100.0%|100% done 
    ```

    The desired model is downloaded.


    **Note:** The model is tuned for the current hardware: 

    ```
        prompt% env MIOPEN_FIND_ENFORCE=3 migraphx-driver run resnet50_fp32.mxr

    ```

4. Run a MIGraphX example using a downloaded model.

**Note:** This example is adapted from [Performing Inference using MIGraphX Python Library](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/develop/examples/vision/python_resnet50) in the ROCm™ software platform documentation. For more details, refer to [`resnet50_inference.ipynb`](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/develop/examples/vision/python_resnet50/resnet50_inference.ipynb).

## 4.3.3.2: Prepare the Example

 1. Install the Python packages used in the example.

    ```
        prompt% pip install opencv-python==4.1.2.30 

        prompt% pip install matplotlib 

    ```

2. Clone the MIGraphX repository to get the example.

    ```
        prompt% cd ~ 

        prompt% git clone https://github.com/ROCmSoftwarePlatform/AMDMIGraphX 

        prompt% cd AMDMIGraphX/examples/vision/python_resnet50 
    ```

3. Download a sample video and name it sample_vid.mp4.

    ```

        prompt% apt install youtube-dl 

        prompt% youtube-dl https://youtu.be/TkqYmvH_XVs 

        prompt% mv sample_vid-TkqYmvH_XVs.mp4 sample_vid.mp4 

    ```
    

## 4.3.3.3: Run the Example from Python
    
```
    prompt% python3 

        import numpy as np 

        from matplotlib import pyplot as plt 

        import cv2 

        import json 

        import time 

        import os.path 

        from os import path 

        import sys  

        import migraphx  

        with open(‘imagenet_simple_labels.json’) as json_data: 

        labels = json.load(json.data) 

```



## 4.3.3.4: Set Up the Video and Capture the Model

```

    model = migraphx.parse_onnx("resnet50_fp32.mxr") 

    model.compile(migraphx.get_target("gpu")) 

    model.print()     # Printed in terminal  

    cap = cv2.VideoCapture("sample_vid.mp4") 

```


## 4.3.3.5: Add Code for Preprocessing Video Frames


```
    def make_nxn(image, n): 

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

    if height > width: 

        dif = height - width 

        bar = dif // 2  

        square = image[(bar + (dif % 2)):(height - bar),:] 

        return cv2.resize(square, (n, n)) 

    elif width > height: 

        dif = width - height 

        bar = dif // 2 

        square = image[:,(bar + (dif % 2)):(width - bar)] 

        return cv2.resize(square, (n, n)) 

    else: 

        return cv2.resize(image, (n, n))         

    def preprocess(img_data): 

        mean_vec = np.array([0.485, 0.456, 0.406]) 

        stddev_vec = np.array([0.229, 0.224, 0.225]) 

        norm_img_data = np.zeros(img_data.shape).astype('float32') 

        for i in range(img_data.shape[0]):   

            norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i] 

        return norm_img_data  

    def predict_class(frame) -> int: 

        # Crop and resize original image 

        cropped = make_nxn(frame, 224) 

        # Convert from HWC to CHW 

        chw = cropped.transpose(2,0,1) 

    # Apply normalization 

        pp = preprocess(chw) 

    # Add singleton dimension (CHW to NCHW) 

        data = np.expand_dims(pp.astype('float32'),0) 

    # Run the model 

        results = model.run({'data':data}) 

    # Extract the index of the top prediction 

        res_npa = np.array(results[0]) 

        return np.argmax(res_npa) 

```

## 4.3.3.6: Run the Complete Look over Video

```

    while (cap.isOpened()): 

        start = time.perf_counter() 

        ret, frame = cap.read() 

        if not ret: break      

        top_prediction = predict_class(frame)      

        end = time.perf_counter() 

        fps = 1 / (end - start) 

        fps_str = f"Frames per second: {fps:0.1f}" 

        label_str = "Top prediction: {}".format(labels[top_prediction])  

        labeled = cv2.putText(frame,  

                          label_str,  

                          (50, 50),  

                          cv2.FONT_HERSHEY_SIMPLEX,  

                          2,  

                          (255, 255, 255),  

                          3,  

                          cv2.LINE_AA) 

    labeled = cv2.putText(labeled,  

                          fps_str,  

                          (50, 1060),  

                          cv2.FONT_HERSHEY_SIMPLEX,  

                          2,  

                          (255, 255, 255),  

                          3,  

                          cv2.LINE_AA) 

    cv2.imshow("Resnet50 Inference", labeled)  

    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' to quit 

        break  

    cap.release() 

    cv2.destroyAllWindows() 
```

<hr/>

[< Previous](/docs/4_deploy_your_own_model/quantize_model/quantizemodel.md) | [Next >](/docs/4_deploy_your_own_model/serve_model/servingmodelwithinferenceserver.md)

<hr/>


# License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.
