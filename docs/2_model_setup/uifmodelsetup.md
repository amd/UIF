<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.2 User Guide </h1>
    </td>
 </tr>
 <tr>-
 <td align="center"><h1>Step 2.0: UIF Model Setup</h1>
 </td>
 </tr>
</table>

# Table of Contents

- [2.1: UIF Model Zoo Introduction](#21-uif-model-zoo-introduction)
	- [2.1.1: Standard Naming Rules](#211-standard-naming-rules)
	- [2.1.2: Model List](#212-model-list)
    - [2.1.3: Once-For-All: Efficient Model Customization for Various Platforms](#213-once-for-all-ofa-efficient-model-customization-for-various-platforms)
- [2.2: Get MIGraphX Models from UIF Model Zoo](#22-get-migraphx-models-from-uif-model-zoo)
- [2.3: Set Up MIGraphX YModel](#23-set-up-migraphx-ymodel)
- [2.4: Get Vitis AI Models from UIF Model Zoo](#24-get-vitis-ai-models-from-uif-model-zoo)
- [2.5: Get ZenDNN Models from UIF Model Zoo](#25-get-zendnn-models-from-uif-model-zoo)

  _Click [here](/README.md#implementing-uif-12) to go back to the UIF User Guide home page._

# 2.1: UIF Model Zoo Introduction

UIF 1.2 Model Zoo provides 50 models for AMD Instinct™ GPUs (MIGraphX) including 20 new models and 30 models inherited from UIF 1.1. The Vitis™ AI development environment provides 106 reference models for different FPGA adaptive platforms. 
Also, you could use the 84 models for AMD EPYC™ CPUs (ZenDNN) inherited from UIF1.1.

Model information is located in [model-list](/docs/2_model_setup/model-list). 

**Note:** If a model is marked as limited to non-commercial use, you must comply with the [AMD license agreement for non-commercial models](/docs/2_model_setup/AMD-license-agreement-for-non-commercial-models.md). 


**UIF 1.2 Models for MIGraphX**
	
<details>
 <summary>Click here to view details</summary>	
	
| #    | Model                   | Original Platform | Datatype FP32 | Datatype FP16 | Pruned | Reminder for limited use scope |
| ---- | ----------------------- | ----------------- | ------------- | ------------- | ------ | ------------------------------ |
| 1    | 2D-Unet                 | TensorFlow2       | √             | √             | ×      |                                |
| 2    | 2D-Unet pruned0.7       | TensorFlow2       | √             | √             | √      |                                |
| 3    | Albert-base             | PyTorch           | √             | √             | ×      |                                |
| 4    | Albert-large            | PyTorch           | √             | √             | ×      |                                |
| 5    | Bert-base               | TensorFlow2       | √             | √             | ×      |                                |
| 6    | Bert-large              | TensorFlow2       | √             | √             | ×      |                                |
| 7    | DETR                    | PyTorch           | √             | √             | ×      |                                |
| 8    | DistillBert             | PyTorch           | √             | √             | ×      |                                |
| 9    | DLRM (40M)              | PyTorch           | √             | √             | ×      | Non-Commercial Use Only        |
| 10   | EfficientDet            | TensorFlow2       | √             | √             | ×      |                                |
| 11   | GPT2-large              | PyTorch           | √             | √             | ×      |                                |
| 12   | GPT2-medium             | PyTorch           | √             | √             | ×      |                                |
| 13   | GPT2-XL                 | PyTorch           | √             | √             | ×      |                                |
| 14   | MobileBert              | PyTorch           | √             | √             | ×      |                                |
| 15   | PointPillars            | PyTorch           | √             | √             | ×      | Non-Commercial Use Only        |
| 16   | Resnet50_v1.5 ofa       | PyTorch           | √             | √             | √      | Non-Commercial Use Only        |
| 17   | RetinaNet               | PyTorch           | √             | √             | ×      |                                |
| 18   | ViT                     | PyTorch           | √             | √             | ×      | Non-Commercial Use Only        |
| 19   | W&D                     | PyTorch           | √             | √             | ×      |                                |
| 20   | yolov3                  | TensorFlow2       | √             | √             | ×      |                                |
	
</details>	
	

**UIF 1.2 Models for Vitis AI**

<details>
 <summary>Click here to view details</summary>
	
| #    | Model                        | Platform   | Datatype FP32 | Datatype INT8 | Pruned | Reminder for limited use scope |
| ---- | :--------------------------- | :--------- | :-----------: | :-----------: | :----: | ------------------------------ |
| 1    | inceptionv1                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 2    | inceptionv1 pruned0.09       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 3    | inceptionv1 pruned0.16       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 4    | inceptionv3                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 5    | inceptionv3 pruned0.2        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 6    | inceptionv3 pruned0.4        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 7    | inceptionv4                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 8    | inceptionv4 pruned0.2        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 9    | inceptionv4 pruned0.4        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 10   | mobilenetv1_0.25             | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 11   | mobilenetv1_1.0              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 12   | mobilenetv1_1.0 pruned0.11   | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 13   | mobilenetv1_1.0 pruned0.12   | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 14   | mobilenetv2_1.0              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 15   | mobilenetv2_1.4              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 16   | resnetv1_50                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 17   | resnetv1_50 pruned0.38       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 18   | resnetv1_50 pruned0.65       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 19   | resnetv1_101                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 20   | resnetv1_101 pruned0.35      | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 21   | resnetv1_101 pruned0.57      | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 22   | resnetv1_152                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 23   | resnetv1_152 pruned0.51      | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 24   | resnetv1_152pruned0.60       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 25   | vgg16                        | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 26   | vgg16 pruned0.43             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 27   | vgg16 pruned0.50             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 28   | vgg19                        | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 29   | vgg19 pruned0.24             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 30   | vgg19 pruned0.39             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 31   | resnetv2_50                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 32   | resnetv2_101                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 33   | resnetv2_152                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 34   | efficientnet-edgetpu-S       | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 35   | efficientnet-edgetpu-M       | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 36   | efficientnet-edgetpu-L       | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 37   | mlperf_resnet50              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 38   | resnet50                     | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 39   | mobilenetv1                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 40   | inceptionv3                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 41   | efficientnet-b0              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 42   | mobilenetv3                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 43   | efficientnet-lite            | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 44   | ViT                          | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 45   | ssdmobilenetv1               | TensorFlow |       √       |       √       |   ×    |                                |
| 46   | ssdmobilenetv2               | TensorFlow |       √       |       √       |   ×    |                                |
| 47   | yolov3                       | TensorFlow |       √       |       √       |   ×    |                                |
| 48   | mlperf_resnet34              | TensorFlow |       √       |       √       |   ×    |                                |
| 49   | efficientdet-d2              | TensorFlow |       √       |       √       |   ×    |                                |
| 50   | yolov3                       | TensorFlow |       √       |       √       |   ×    |                                |
| 51   | yolov4_416                   | TensorFlow |       √       |       √       |   ×    |                                |
| 52   | yolov4_512                   | TensorFlow |       √       |       √       |   ×    |                                |
| 53   | RefineDet-Medical            | TensorFlow |       √       |       √       |   ×    |                                |
| 54   | RefineDet-Medical pruned0.50 | TensorFlow |       √       |       √       |   √    |                                |
| 55   | RefineDet-Medical pruned0.75 | TensorFlow |       √       |       √       |   √    |                                |
| 56   | RefineDet-Medical pruned0.85 | TensorFlow |       √       |       √       |   √    |                                |
| 57   | RefineDet-Medical pruned0.88 | TensorFlow |       √       |       √       |   √    |                                |
| 58   | bert-base                    | TensorFlow |       √       |       √       |   ×    |                                |
| 59   | superpoint                   | TensorFlow |       √       |       √       |   ×    |                                |
| 60   | HFNet                        | TensorFlow |       √       |       √       |   ×    |                                |
| 61   | rcan                         | TensorFlow |       √       |       √       |   ×    |                                |
| 62   | inceptionv3                  | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 63   | inceptionv3 pruned0.3        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 64   | inceptionv3 pruned0.4        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 65   | inceptionv3 pruned0.5        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 66   | inceptionv3 pruned0.6        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 67   | squeezenet                   | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 68   | resnet50_v1.5                | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 69   | resnet50_v1.5 pruned0.3      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 70   | resnet50_v1.5 pruned0.4      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 71   | resnet50_v1.5 pruned0.5      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 72   | resnet50_v1.5 pruned0.6      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 73   | resnet50_v1.5 pruned0.7      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 74   | OFA-resnet50                 | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 75   | OFA-resnet50 pruned0.45      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 76   | OFA-resnet50 pruned0.60      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 77   | OFA-resnet50 pruned0.74      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 78   | OFA-resnet50 pruned0.88      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 79   | OFA-depthwise-resnet50       | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 80   | vehicle type classification  | PyTorch    |       √       |       √       |   ×    |                                |
| 81   | vehicle make classification  | PyTorch    |       √       |       √       |   ×    |                                |
| 82   | vehicle color classification | PyTorch    |       √       |       √       |   ×    |                                |
| 83   | OFA-yolo                     | PyTorch    |       √       |       √       |   ×    |                                |
| 84   | OFA-yolo pruned0.3           | PyTorch    |       √       |       √       |   √    |                                |
| 85   | OFA-yolo pruned0.6           | PyTorch    |       √       |       √       |   √    |                                |
| 86   | yolox-nano                   | PyTorch    |       √       |       √       |   ×    |                                |
| 87   | yolov4csp                    | PyTorch    |       √       |       √       |   ×    |                                |
| 88   | yolov6m                      | PyTorch    |       √       |       √       |   ×    |                                |
| 89   | pointpillars                 | PyTorch    |       √       |       √       |   ×    |                                |
| 90   | HRNet                        | PyTorch    |       √       |       √       |   ×    |                                |
| 91   | 3D-UNET                      | PyTorch    |       √       |       √       |   ×    |                                |
| 92   | bert-base                    | PyTorch    |       √       |       √       |   ×    |                                |
| 93   | bert-large                   | PyTorch    |       √       |       √       |   ×    |                                |
| 94   | bert-tiny                    | PyTorch    |       √       |       √       |   ×    |                                |
| 95   | face-mask-detection          | PyTorch    |       √       |       √       |   ×    |                                |
| 96   | movenet                      | PyTorch    |       √       |       √       |   ×    |                                |
| 97   | fadnet                       | PyTorch    |       √       |       √       |   ×    |                                |
| 98   | fadnet pruned0.65            | PyTorch    |       √       |       √       |   √    |                                |
| 99   | fadnetv2                     | PyTorch    |       √       |       √       |   ×    |                                |
| 100  | fadnetv2 pruned0.51          | PyTorch    |       √       |       √       |   √    |                                |
| 101  | psmnet pruned0.68            | PyTorch    |       √       |       √       |   √    |                                |
| 102  | SESR-S                       | PyTorch    |       √       |       √       |   ×    |                                |
| 103  | OFA-rcan                     | PyTorch    |       √       |       √       |   ×    |                                |
| 104  | xilinxSR                     | PyTorch    |       √       |       √       |   ×    |                                |
| 105  | yolov7                       | PyTorch    |       √       |       √       |   ×    |                                |
| 106  | 2D-UNET                      | PyTorch    |       √       |       √       |   ×    |                                |

</details>

**UIF 1.1 Models for ZenDNN**
	
<details>
 <summary>Click here to view details</summary>	

| #    | Model                   | Original Platform | Converted Format | Datatype FP32 | Datatype BF16 | Datatype INT8 | Pruned | Reminder for limited use scope |
| ---- | ----------------------- | ----------------- | ---------------- | ------------- | ------------- | ------------- | ------ | ------------------------------ |
| 1    | RefineDet               | TensorFlow        | .PB              | √             | ×             | √             | ×      |                                |
| 2    | RefineDet pruned0.5     | TensorFlow        | .PB              | √             | ×             | √             | √      |                                |
| 3    | RefineDet pruned0.75    | TensorFlow        | .PB              | √             | ×             | √             | √      |                                |
| 4    | RefineDet pruned0.85    | TensorFlow        | .PB              | √             | ×             | √             | √      |                                |
| 5    | RefineDet pruned0.88    | TensorFlow        | .PB              | √             | ×             | √             | √      |                                |
| 6    | EfficientNet-L          | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 7    | EfficientNet-M          | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 8    | EfficientNet-S          | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 9    | Resnetv1_50             | TensorFlow        | .PB              | √             | √             | √             | ×      | Non-Commercial Use Only        |
| 10   | Resnetv1_50 pruned0.38  | TensorFlow        | .PB              | √             | √             | √             | √      | Non-Commercial Use Only        |
| 11   | Resnetv1_50 pruned0.65  | TensorFlow        | .PB              | √             | √             | √             | √      | Non-Commercial Use Only        |
| 12   | Resnetv1_101            | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 13   | Resnetv1_101 pruned0.34 | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 14   | Resnetv1_101 pruned0.57 | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 15   | Resnetv1_152            | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 16   | Resnetv1_152 pruned0.51 | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 17   | Resnetv1_152 pruned0.60 | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 18   | Inceptionv1             | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 19   | Inceptionv1 pruned0.09  | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 20   | Inceptionv1 pruned0.16  | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 21   | Inceptionv3             | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 22   | Inceptionv3 pruned0.2   | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 23   | Inceptionv3 pruned0.4   | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 24   | Inceptionv4             | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 25   | Inceptionv4 pruned0.2   | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 26   | Inceptionv4 pruned0.4   | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 27   | Mobilenetv1             | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 28   | Mobilenetv1 pruned0.11  | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 29   | Mobilenetv1 pruned0.12  | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 30   | VGG16                   | TensorFlow        | .PB              | √             | √             | √             | ×      | Non-Commercial Use Only        |
| 31   | VGG16 pruned0.43        | TensorFlow        | .PB              | √             | √             | √             | √      | Non-Commercial Use Only        |
| 32   | VGG16 pruned0.50        | TensorFlow        | .PB              | √             | √             | √             | √      | Non-Commercial Use Only        |
| 33   | VGG19                   | TensorFlow        | .PB              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 34   | VGG19 pruned0.24        | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 35   | VGG19 pruned0.39        | TensorFlow        | .PB              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 36   | Bert-base               | TensorFlow        | .PB              | √             | ×             | ×             | ×      |                                |
| 37   | Albert-base             | TensorFlow        | .PB              | √             | ×             | ×             | ×      |                                |
| 38   | Albert-large            | TensorFlow        | .PB              | √             | ×             | ×             | ×      |                                |
| 39   | Mobilebert              | TensorFlow        | .PB              | √             | ×             | ×             | ×      |                                |
| 40   | Resnet50_v1.5           | PyTorch           | .PT              | √             | √             | √             | ×      | Non-Commercial Use Only        |
| 41   | Resnet50_v1.5 pruned0.3 | PyTorch           | .PT              | √             | √             | √             | √      | Non-Commercial Use Only        |
| 42   | Resnet50_v1.5 pruned0.4 | PyTorch           | .PT              | √             | √             | √             | √      | Non-Commercial Use Only        |
| 43   | Resnet50_v1.5 pruned0.5 | PyTorch           | .PT              | √             | √             | √             | √      | Non-Commercial Use Only        |
| 44   | Resnet50_v1.5 pruned0.6 | PyTorch           | .PT              | √             | √             | √             | √      | Non-Commercial Use Only        |
| 45   | Resnet50_v1.5 pruned0.7 | PyTorch           | .PT              | √             | √             | √             | √      | Non-Commercial Use Only        |
| 46   | Inceptionv3             | PyTorch           | .PT              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 47   | Inceptionv3 pruned0.3   | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 48   | Inceptionv3 pruned0.4   | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 49   | Inceptionv3 pruned0.5   | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 50   | Inceptionv3 pruned0.6   | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 51   | Reid_resnet50           | PyTorch           | .PT              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 52   | Reid_resnet50 pruned0.4 | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 53   | Reid_resnet50 pruned0.5 | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 54   | Reid_resnet50 pruned0.6 | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 55   | Reid_resnet50 pruned0.7 | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 56   | OFA_resnet50            | PyTorch           | .PT              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 57   | OFA_resnet50 pruned0.45 | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 58   | OFA_resnet50 pruned0.60 | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 59   | OFA_resnet50 pruned0.74 | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 60   | OFA_resnet50 pruned0.88 | PyTorch           | .PT              | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 61   | 3D-Unet                 | PyTorch           | .PT              | √             | ×             | ×             | ×      |                                |
| 62   | 3D-Unet pruned0.1       | PyTorch           | .PT              | √             | ×             | ×             | √      |                                |
| 63   | 3D-Unet pruned0.2       | PyTorch           | .PT              | √             | ×             | ×             | √      |                                |
| 64   | 3D-Unet pruned0.3       | PyTorch           | .PT              | √             | ×             | ×             | √      |                                |
| 65   | 3D-Unet pruned0.4       | PyTorch           | .PT              | √             | ×             | ×             | √      |                                |
| 66   | 3D-Unet pruned0.5       | PyTorch           | .PT              | √             | ×             | ×             | √      |                                |
| 67   | DLRM                    | PyTorch           | .PT              | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 68   | GPT-2 small             | PyTorch           | .PT              | √             | ×             | ×             | ×      |                                |
| 69   | DistilGPT-2             | PyTorch           | .PT              | √             | ×             | ×             | ×      |                                |
| 70   | Albert-base             | PyTorch           | .PT              | √             | ×             | ×             | ×      |                                |
| 71   | Albert-large            | PyTorch           | .PT              | √             | ×             | ×             | ×      |                                |
| 72   | Mobilebert              | PyTorch           | .PT              | √             | ×             | ×             | ×      |                                |
| 73   | Resnetv1_50             | ONNXRT            | .ONNX            | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 74   | Resnetv1_50 pruned0.38  | ONNXRT            | .ONNX            | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 75   | Resnetv1_50 pruned0.65  | ONNXRT            | .ONNX            | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 76   | Inceptionv3             | ONNXRT            | .ONNX            | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 77   | Inceptionv3 pruned0.2   | ONNXRT            | .ONNX            | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 78   | Inceptionv3 pruned0.4   | ONNXRT            | .ONNX            | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 79   | Mobilenetv1             | ONNXRT            | .ONNX            | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 80   | Mobilenetv1 pruned0.11  | ONNXRT            | .ONNX            | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 81   | Mobilenetv1 pruned0.12  | ONNXRT            | .ONNX            | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 82   | VGG16                   | ONNXRT            | .ONNX            | √             | ×             | √             | ×      | Non-Commercial Use Only        |
| 83   | VGG16 pruned0.43        | ONNXRT            | .ONNX            | √             | ×             | √             | √      | Non-Commercial Use Only        |
| 84   | VGG16 pruned0.50        | ONNXRT            | .ONNX            | √             | ×             | √             | √      | Non-Commercial Use Only        |

</details>	


### 2.1.1: Standard Naming Rules

Model name: `F_M_(P)_V_Z`
* `F` specifies the training framework: `tf` is TensorFlow 1.x, `tf2` is TensorFlow 2.x, `pt` is PyTorch, `onnx` is ONNXRT.
* `M` specifies the model.
* `P` specifies the pruning ratio, meaning how much computation is reduced. It is optional depending on whether the model is pruned or not.
* `V` specifies the version of UIF.
* `Z` specifies the version of ZenDNN or MIGraphX.

For example, `pt_resnet50v1.5_pruned0.6_1.2_M2.6` is a `resnet50 v1.5` model trained with `PyTorch`, `60%` pruned, the UIF release version is `1.2`, and the MIGraphX version is `2.6`.

### 2.1.2: Model List

Visit [model-list](/docs/2_model_setup/model-list) The models are named according to standard naming rules. 

You can get a download link and MD5 checksum of all released models running on different hardware platforms from here. You can download it manually or use the the following automatic download script to search for models by keyword.   

### 2.1.3: Once-For-All (OFA): Efficient Model Customization for Various Platforms

Once-For-All (OFA) is an efficient neural architecture search (NAS) method that enables the customization of sub-networks for various hardware platforms. It decouples training and search, enabling quick derivation of specialized models optimized for specific platforms with efficient inference performance. OFA offers the following benefits:

1. OFA requires only one training process and can specialize in diverse hardware platforms (for example, DPU, GPU, CPU, IPU) without incurring the heavy computation costs associated with manual design or conventional RL-based NAS methods.

2. By decoupling supernet training and subnetwork searching, OFA optimizes networks with actual latency constraints of a specific hardware platform, avoiding discrepancies between estimated and actual latencies.

3. OFA offers strong flexibility and scalability regarding the search space, supporting various network architectures (for example, CNN and Transformer) with different granularities (for example, operation-wise, layer-wise, block-wise) to cater to different tasks (for example, CV, NLP). 

4. Extensive experiments with OFA demonstrate consistent performance acceleration while maintaining similar accuracy levels compared to baselines on different devices. For instance, OFA-ResNet achieves a speedup ratio of 69% on MI100, 78% on MI210, and 97% on Navi2 compared to ResNet50 baselines with a batch size of 1 and prune ratio of 78%.

5. OFA has the potential to search for optimized large language models (LLM) by stitching pretrained models for accuracy-efficiency trade-offs. A proof of concept for Vision Transformer (ViT) shows that OFA can derive an optimized ViT with a 20% to 40% speedup ratio compared to ViT-base on MI100.

#### 2.1.3.1: OFA-ResNet for AMD GPUs with MIGraphX

- Task Description: This case aims to search for ResNet-like models optimized for AMD GPUs with MIGraphX. The latency on MI100 is used as constraints for optimization. Channel numbers are aligned with GPU capabilities.
- Search Space Design: The search space has the following parameters:
  - Stage output ratio (The ratio of output channel for each stage of the model): [0.65, 0.8, 1.0]
  - Depth (The number of blocks for each stage): [0, 1, 2]
  - Expand ratio (The ratio of output channel for each block in each stage): [0.2, 0.25, 0.35]
  - Resolution (The input size of the model): [128, 160, 192, 224]
- Results: The OFA-ResNet model achieves a speedup ratio of 69% on MI100, 78% on MI210, 97% on Navi2 compared to ResNet50 baseline (with batch size of 1 and pruned ratio of 78%) while maintaining similar accuracy.

#### 2.1.3.2: Performance Comparison

| Model       | Float Accuracy (ImageNet 1K) | FLOPs (G) | Pruned Ratio | Speedup ratio on MI100 | Speedup ratio on MI210 | Speedup ratio on Navi2 |
|-------------|------------------------------|-----------|--------------|------------------------|------------------------|------------------------|
| ResNet50    | 76.1%                        | 8.2       | 0%           | -                      | -                      | -                      |
| OFA-ResNet  | 75.8%                        | 1.77      | 78%          | 1.69x                  | 1.78x                  | 1.97x                  |


# 2.2: Get MIGraphX Models from UIF Model Zoo

Perform the following steps to install UIF 1.2 models:

1. Set up and run the model downloader tool.

    ```
    git clone https://github.com/AMD/uif.git
    cd uif/docs/2_model_setup
    python3 downloader.py
    ```
    It has the following provision to specify the frameworks:

    ```
    Tip:
    You need to input framework and model name. Use space divide such as tf vgg16
    tf:tensorflow1.x  tf2:tensorflow2.x  onnx:onnxruntime  dk:darknet  pt:pytorch  all: list all model
    input:
    ```
2. Download UIF 1.2 GPU models. Provide `pt` as input to get the list of models.

    ```
	input:pt
	chose model
	0 : all
    1 : pt_inceptionv3_1.2_M2.6
    2 : pt_bert_base_1.2_M2.6
    3 : pt_bert_large_1.2_M2.6
    

	...
	input num:
    ```

The models with 1.2 as suffix are UIF 1.2 models. MI100 means the model has been tuned for MI-100 GPU, and MI210 indicates the model has been tuned for MI-210 GPU.  Without either of these suffixes, the model is a model version that should be used for training, including the GPU example later in this section.

    ```
	input num:1
	chose model type
	0: all
	1: GPU
	2: MI100
	3: MI200
	...
	...
    input num:
    
    ```
    
Provide `1` as input to download the GPU model.

    ```
    input num:1
    pt_inceptionv3_1.2_M2.6.zip
	                                              100.0%|100%
	done
    ```
**Note**
See *pt_resnet50v1.5_0.4_1.1_M2.6.tar.gz* for the zipped file.

# 2.3: Set Up MIGraphX YModel

YModel is designed to provide significant inference performance through MIGraphX. Prior to the introduction of YModel, performance tuning was conditioned by having tuned kernel configs stored in a `/home` local User DB. If users move their model to a different server or allow a different user to use it, they would have to run through the MIOpen tuning process again to populate the next User DB with the best kernel configs and corresponding solvers. Tuning is time consuming, and if the users have not performed tuning, they would see discrepancies between expected or claimed inference performance and the actual inference performance. This leads to repetitive and time-consuming tuning tasks for each user. 

MIGraphX introduces a feature known as YModel that stores the kernel config parameters found during tuning in the same file as the model itself. This ensures the same level of expected performance even when a model is copied to a different user or system.   

**Note:** The YModel feature is available starting with the ROCm™ v5.4.1 and UIF v1.1 releases. For more information on ROCm and the YModel feature, see https://rocm.docs.amd.com/en/latest/examples/machine_learning/migraphx_optimization.html#ymodel. 

**Note:** YModel does not support MIOpen fusions and must be disabled while generating YModel.

To set up the YModel for GPUs:

1. Tune the kernels for your architecture with `MIOPEN_FIND_ENFORCE=3 migraphx-driver run <file.onnx>`.

2. Build a `*.mxr` file with a compile option: 

```
	migraphx-driver compile file.onnx --enable-offload-copy --gpu --binary -o file.mxr
	
```

3. Run the (Python) program with a command line: 

```
	model=migraphx.load("file.mxr")
```

**Note:** If the Model Zoo contains a `*.mxr` file for your architecture, you can skip steps 1 and 2. 

# 2.4: Get Vitis AI Models from UIF Model Zoo

Follow the instructions in the [Vitis AI](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo) Model Zoo page.


# 2.5: Get ZenDNN Models from UIF Model Zoo

Perform the following steps to install UIF 1.1 models:

1. Set up and run the model downloader tool.

    ```
    git clone https://github.com/AMD/uif.git
    cd uif/docs/2_model_setup
    python3 downloader.py
    ```
    It has the following provision to specify the frameworks:

    ```
    Tip:
    You need to input framework and model name, use space divide such as tf vgg16
    tf:tensorflow1.x  tf2:tensorflow2.x  onnx:onnxruntime  dk:darknet  pt:pytorch  all: list all model
    input:
    ```
2. Download UIF 1.1 TensorFlow+ZenDNN models. Provide `tf` as input to get the list of models.

    ```
	input:tf
	chose model
	0 : all
	1 : tf_refinedet_edd_320_320_81.28G_1.1_Z4.0
	2 : tf_refinedet_edd_320_320_0.5_41.42G_1.1_Z4.0
	3 : tf_refinedet_edd_320_320_0.75_20.54G_1.1_Z4.0
	4 : tf_refinedet_edd_320_320_0.85_12.32G_1.1_Z4.0
    ...
	...
	input num:
    ```
    The models with 1.1 as suffix are UIF 1.1 models. Z4.0 means ZenDNN version is `4.0`. Provide the model number corresponding to the required model as input. 
    For example, select `1` for a RefineDet baseline model.

    ```
	input num:1
	chose model type
	0: all
	1 : CPU
	
	...
	...
    input num:
    ```
    Provide `1` as input to download the ZenDNN CPU model.

    ```
	input num:1
    tf_refinedet_edd_320_320_81.28G_1.1_Z4.0.zip
	                                              100.0%|100%
	done
    ```
3. Download UIF 1.1 PyTorch+ZenDNN models. Provide `pt` as input to get the list of models.

    ```
	input:pt
	chose model
	0 : all
    1 : pt_inceptionv3_imagenet_299_299_11.4G_1.1_M2.4
    2 : pt_3dunet_kits19_128_128_128_0.3_763.8G_1.1_Z4.0
    3 : pt_bert_base_SQuADv1.1_384_70.66G_1.1_M2.4
    4 : pt_bert_large_SQuADv1.1_384_246.42G_1.1_M2.4
    5 : pt_personreid-res50_market1501_256_128_0.4_3.3G_1.1_Z4.0
    ...
	...
	input num:
    ```
    Models with 1.1 as suffix are UIF 1.1 models. Provide the model number corresponding to the required model as input. For example, select `3` for resnet50 pruned 40% model.

    ```
	input num:2
	chose model type
	0: all
	1 : CPU
	
	...
	...
    input num:
    ```
    Provide `1` as input to download the UIF 1.1 ZenDNN CPU model.
    ```
	input num:1
    pt_3dunet_kits19_128_128_128_0.3_763.8G_1.1_Z4.0.zip
	                                              100.0%|100%
	done
    ```

4. Download UIF 1.1 ONNXRT+ZenDNN models. Provide `onnx` as input to get the list of models.

    ```
	input:onnx
    chose model
    0 : all
    1 : onnx_vgg16_imagenet_224_224_30.96G_1.1_Z4.0
    2 : onnx_inceptionv3_imagenet_299_299_0.4_6.9G_1.1_Z4.0
    3 : onnx_resnetv1_50_imagenet_224_224_0.38_4.3G_1.1_Z4.0
    4 : onnx_mobilenetv1_1.0_imagenet_224_224_1.14G_1.1_Z4.0
    ...
	...
	input num:
    ```
    Models with 1.1 as suffix are UIF 1.1 models. Provide the model number corresponding to the required model as input. For example, select `3` for resnet50 pruned 40% model.

    ```
	input num:3
	chose model type
	0: all
	1 : CPU
	
	...
	...
    input num:
    ```
    Provide `1` as input to download the UIF 1.1 ZenDNN CPU model.
    ```
	input num:1
    onnx_resnetv1_50_imagenet_224_224_0.38_4.3G_1.1_Z4.0.zip
	                                              100.0%|100%
	done
    ```

<hr/>

[< Previous](/docs/1_installation/installation.md) | [Next >](/docs/2_model_setup/gpu_model_example.md)

<hr/>

 # License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.

