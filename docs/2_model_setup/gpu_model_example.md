<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.1 User Guide </h1>
    </td>
 </tr>
 <tr>
 <td align="center"><h1>Step 2.6: GPU Model Example</h1>
 </td>
 </tr>
</table>

# Table of Contents 

- [2.6.1: Installation](#261-Installation)
- [2.6.2: Data Preparation](#262-Data-Preparation)
- [2.6.3: Training and Evaluation](#263-Training-and-Evaluation)
- [2.6.4: Performance](#264-Performance)
- [2.6.5: Data Preprocessing for Inference](#265-Data-Preprocessing-for-Inference)

  _Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._

UIF accelerates deep learning inference applications on all AMD compute platforms for popular machine learning frameworks, including TensorFlow, PyTorch, and ONNXRT. UIF 1.1 extends the support to AMD Instinct™ GPUs. Currently, [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX) is the acceleration library for Deep Learning Inference running on AMD Instinct GPUs. 

The following example takes a PyTorch ResNet-50-v1.5 model selected from UIF Model Zoo as an example to show how it works on different GPU platforms.

**Note:**  The model tuning time on a MI210 device is long (around three hours). With MI210, it is recommended to skip this step and use the YModel provided in the model packages.

# 2.6.1: Installation
 
After unzipping the PyTorch ResNet50-v1.5 model, you need to set environment and install dependencies. It is recommended that you use Docker® software for training and testing.

  1. Download the UIF Docker image with the following instruction:
     
    docker pull amdih/uif-pytorch:uif1.1_rocm5.4.1_vai3.0_py3.7_pytorch1.12
    

   **Note**: AMD GPU dependencies and PyTorch are pre-installed in the Docker. 
        
  2. Launch a container from the Docker image. 

  3. Install the Python dependencies using `pip`:

   ```shell
  $ pip install --user -r requirements.txt
   ```


# 2.6.2: Data Preparation

Next, use ImageNet to retrain and test this ResNet50 model.

For the ImageNet dataset, go to [ImageNet](http://image-net.org/download-images). The downloaded ImageNet dataset needs to be organized in the following format:

1. Create a folder of "data" . Put the validation data in `+data/val`.
2. Data from each class for validation set needs to be put in the folder:

  ```
    +data/Imagenet/val
         +val/n01847000
         +val/n02277742
         +val/n02808304
         +...
  ```
  
# 2.6.3: Training and Evaluation

1. Train ResNet50 on ImageNet from scratch.

The pretrained ResNet50 model is provided in the ``float`` folder. If you want to evaluate directly, skip this step. 

```
  $ cd code
  $ sh run_train.sh
  
  
Training...
Used arguments: Namespace(data_dir='data/imagenet/', deployable='./deployable.pth', display_freq=100, early_stop=False, epochs=100, lr=0.01, mode='train', pretrained=None, prune_ratio=0, quantizer_norm=True, save_dir='./save', train_batch_size=256, val_batch_size=64, val_freq=1000, warmup_epochs=5, weight_decay=0.0001, weight_lr_decay=0.94, workers=4)
Fri Dec  9 14:26:00 2022 , Learning rate: 3.9960039960039963e-07
Epoch[0], Step: [     0/500500] Time  2.094 ( 2.094)    Data  0.000 ( 0.000)    Loss 7.0611e+00 (7.0611e+00)    Acc@1   0.00 (  0.00)   Acc@5   0.39 (  0.39)
Epoch[0], Step: [   100/500500] Time  0.379 ( 0.431)    Data  0.000 ( 0.000)    Loss 7.0727e+00 (7.0804e+00)    Acc@1   0.00 (  0.09)   Acc@5   0.39 (  0.41)
Epoch[0], Step: [   200/500500] Time  0.647 ( 0.425)    Data  0.000 ( 0.000)    Loss 7.0127e+00 (7.0654e+00)    Acc@1   0.00 (  0.10)   Acc@5   0.39 (  0.43)
...
  ```

2.  Evaluate the accuracy of the FP32 float model.

```
  $ sh run_test_float.sh
  
  
-------- Start resnet50 test
Model name: resnet50 
Model type: pth 
Loading model: float/resnet50_pretrained.pth 
██████████████████████████████████████| 391/391 [22:07<00:00,  3.40s/it] top-1 / top-5 accuracy: 76.1 / 92.9
-------- End of resnet50 test
  ```
  
3. Evaluate accuracy and performance with MIGraphX. Before the evaluation, convert the native PyTorch model (`.pth`) to an FP32/FP16 onnx model. Run the following script to get both FP32 and FP16 onnx models.

 ```
  $ sh export_float_onnx_model.sh
  ```

The onnx models are provided in the `float` folder, so you can choose to skip this step. 

- You can then run inference and evaluation with MIGraphX. Here, scripts are provided to evaluate the model's accuracy with MIGraphX.

```
  $ sh run_test_migraphx.sh
  
  
-------- Start resnet50 test
=== Load pretrained model ===                                                                                                                                                                                                                                            
Model name: resnet50
Model type: onnx                                                                                                                                                                                                                                                         
Loading model: float/resnet50_fp32.onnx
Evaluating onnx model using AMD MIGRAPHX    
100%|██████████████████████████████████████████████████████████████| 50000/50000 [05:41<00:00, 146.24it/s]
top-1 / top-5 accuracy: 76.1 / 92.9
-------- End of resnet50 test                                                                                                                                                                                                                                            

-------- Start resnet50 test
=== Load pretrained model ===                                                                                                                                                                                                                                            
Model name: resnet50
Model type: onnx                                                                                                                                                                                                                                                         
Loading model: float/resnet50_fp16.onnx
Evaluating onnx model using AMD MIGRAPHX    
100%|█████████████████████████████████████████████████████████████| 50000/50000 [05:11<00:00, 160.76it/s]
top-1 / top-5 accuracy: 76.1 / 92.9
-------- End of resnet50 test
  ```
  
- You can get the model's inference performance on AMD GPU with MIGraphX Driver. The default batch size is set to 64 in this script.

```
  $ sh test_perf_migraphx.sh
Compiling ...
Reading: ../float/resnet50_fp32.onnx
module: "main"
main:@0 = check_context::migraphx::version_1::gpu::context -> float_type, {}, {}
main:@1 = hip::hip_allocate_memory[shape=float_type, {154140672}, {1},id=main:scratch] -> float_type, {154140672}, {1}
main:@2 = hip::hip_copy_literal[id=main:@literal:99] -> float_type, {64, 3, 7, 7}, {147, 49, 7, 1}
...
Batch size: 64
Rate: 2245.41/sec
Total time: 28.5025ms
Total instructions time: 29.7867ms
Overhead time: 0.0725389ms, -1.28419ms
Overhead: 0%, -5%

Compiling ...
Reading: ../float/resnet50_fp16.onnx
module: "main"
main:@0 = check_context::migraphx::version_1::gpu::context -> float_type, {}, {}
main:@1 = hip::hip_allocate_memory[shape=float_type, {77070336}, {1},id=main:scratch] -> float_type, {77070336}, {1}
main:@2 = hip::hip_copy_literal[id=main:@literal:99] -> half_type, {64, 3, 7, 7}, {147, 49, 7, 1}
main:@3 = load[offset=0,end=0](main:@1) -> int8_type, {0}, {1}
...
Batch size: 64
Rate: 5176.15/sec
Total time: 12.3644ms
Total instructions time: 14.0585ms
Overhead time: 0.0919278ms, -1.69413ms
Overhead: 1%, -14%

  ```

# 2.6.4: Performance

The accuracy and performance of the FP32/16 onnx model on AMD GPU MI100 (MIGraphX driver 2.4) are evaluated as follows:

|Resnet50 Model |Input Size|FLOPs| Top-1/Top-5 Accuracy, %| Performance Rate, /sec |
|----|---|---|---|---|
|PyTorch model| 224x224 | 8.2G|  76.1/92.9 |  - |
|FP32 onnx model| 224x224 | 8.2G|  76.1/92.9 |  bs=1: 484.147<br> bs=64: 5176.15|
|FP16 onnx model| 224x224 | 8.2G|  76.1/92.9 |  bs=1: 734.367<br> bs=64: 5176.15|


# 2.6.5: Data Preprocessing for Inference

The model used in this example uses the following data preprocessing parameters for inference:

  ```
  data channel order: BGR(0~255)
  resize: short side resize to 256 and keep the aspect ratio
  center crop: 224 * 224
  ```


<hr/>

[< Previous](/docs/2_model_setup/uifmodelsetup.md) | [Next >](/docs/3_run_example/runexample-script.md)

<hr/>

 # License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.

