<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.1 User Guide </h1>
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
- [2.2: Get ZenDNN Models from UIF Model Zoo](#22-get-zendnn-models-from-uif-model-zoo)
- [2.3: Get MIGraphX Models from UIF Model Zoo](#23-get-migraphx-models-from-uif-model-zoo)
- [2.4: Set Up MIGraphX YModel](#24-set-up-migraphx-ymodel)
- [2.5: Get Vitis AI Models from UIF Model Zoo](#25-get-vitis-ai-models-from-uif-model-zoo)

  _Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._

# 2.1: UIF Model Zoo Introduction

UIF 1.1 Model Zoo provides 30 models for AMD Instinct™ GPUs (MIGraphX) and 84 models for AMD EPYC™ CPUs (ZenDNN). In the Vitis™ AI development environment, 130 reference models for different FPGA adaptive platforms are also provided. Refer to the following model lists for details. 

**Note:** If a model is marked as limited to non-commercial use, you must comply with the [AMD license agreement for non-commercial models](/docs/2_model_setup/AMD-license-agreement-for-non-commercial-models.md). 


**UIF 1.1 Models for Vitis AI**

<details>
 <summary>Click here to view details</summary>
	
| #    | Model                        | Platform   | Datatype FP32 | Datatype INT8 | Pruned | Reminder for limited use scope |
| ---- | :--------------------------- | :--------- | :-----------: | :-----------: | :----: | ------------------------------ |
| 1    | inception-resnetv2           | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 2    | inceptionv1                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 3    | inceptionv1 pruned0.09       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 4    | inceptionv1 pruned0.16       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 5    | inceptionv2                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 6    | inceptionv3                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 7    | inceptionv3 pruned0.2        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 8    | inceptionv3 pruned0.4        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 9    | inceptionv4                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 10   | inceptionv4 pruned0.2        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 11   | inceptionv4 pruned0.4        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 12   | mobilenetv1_0.25             | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 13   | mobilenetv1_0.5              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 14   | mobilenetv1_1.0              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 15   | mobilenetv1_1.0 pruned0.11   | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 16   | mobilenetv1_1.0 pruned0.12   | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 17   | mobilenetv2_1.0              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 18   | mobilenetv2_1.4              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 19   | resnetv1_50                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 20   | resnetv1_50 pruned0.38       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 21   | resnetv1_50 pruned0.65       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 22   | resnetv1_101                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 23   | resnetv1_101 pruned0.35      | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 24   | resnetv1_101 pruned0.57      | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 25   | resnetv1_152                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 26   | resnetv1_152 pruned0.51      | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 27   | resnetv1_152pruned0.60       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 28   | vgg16                        | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 29   | vgg16 pruned0.43             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 30   | vgg16 pruned0.50             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 31   | vgg19                        | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 32   | vgg19 pruned0.24             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 33   | vgg19 pruned0.39             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 34   | resnetv2_50                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 35   | resnetv2_101                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 36   | resnetv2_152                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 37   | efficientnet-edgetpu-S       | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 38   | efficientnet-edgetpu-M       | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 39   | efficientnet-edgetpu-L       | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 40   | mlperf_resnet50              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 41   | mobilenetEdge1.0             | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 42   | mobilenetEdge0.75            | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 43   | resnet50                     | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 44   | mobilenetv1                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 45   | inceptionv3                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 46   | efficientnet-b0              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 47   | mobilenetv3                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 48   | efficientnet-lite            | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 49   | ViT                          | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 50   | ssdmobilenetv1               | TensorFlow |       √       |       √       |   ×    |                                |
| 51   | ssdmobilenetv2               | TensorFlow |       √       |       √       |   ×    |                                |
| 52   | ssdresnet50v1_fpn            | TensorFlow |       √       |       √       |   ×    |                                |
| 53   | yolov3                       | TensorFlow |       √       |       √       |   ×    |                                |
| 54   | mlperf_resnet34              | TensorFlow |       √       |       √       |   ×    |                                |
| 55   | ssdlite_mobilenetv2          | TensorFlow |       √       |       √       |   ×    |                                |
| 56   | ssdinceptionv2               | TensorFlow |       √       |       √       |   ×    |                                |
| 57   | refinedet                    | TensorFlow |       √       |       √       |   ×    |                                |
| 58   | efficientdet-d2              | TensorFlow |       √       |       √       |   ×    |                                |
| 59   | yolov3                       | TensorFlow |       √       |       √       |   ×    |                                |
| 60   | yolov4_416                   | TensorFlow |       √       |       √       |   ×    |                                |
| 61   | yolov4_512                   | TensorFlow |       √       |       √       |   ×    |                                |
| 62   | RefineDet-Medical            | TensorFlow |       √       |       √       |   ×    |                                |
| 63   | RefineDet-Medical pruned0.50 | TensorFlow |       √       |       √       |   √    |                                |
| 64   | RefineDet-Medical pruned0.75 | TensorFlow |       √       |       √       |   √    |                                |
| 65   | RefineDet-Medical pruned0.85 | TensorFlow |       √       |       √       |   √    |                                |
| 66   | RefineDet-Medical pruned0.88 | TensorFlow |       √       |       √       |   √    |                                |
| 67   | mobilenetv2 (segmentation)   | TensorFlow |       √       |       √       |   ×    |                                |
| 68   | erfnet                       | TensorFlow |       √       |       √       |   ×    |                                |
| 69   | 2d-unet                      | TensorFlow |       √       |       √       |   ×    |                                |
| 70   | bert-base                    | TensorFlow |       √       |       √       |   ×    |                                |
| 71   | superpoint                   | TensorFlow |       √       |       √       |   ×    |                                |
| 72   | HFNet                        | TensorFlow |       √       |       √       |   ×    |                                |
| 73   | rcan                         | TensorFlow |       √       |       √       |   ×    |                                |
| 74   | inceptionv3                  | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 75   | inceptionv3 pruned0.3        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 76   | inceptionv3 pruned0.4        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 77   | inceptionv3 pruned0.5        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 78   | inceptionv3 pruned0.6        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 79   | squeezenet                   | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 80   | resnet50_v1.5                | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 81   | resnet50_v1.5 pruned0.3      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 82   | resnet50_v1.5 pruned0.4      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 83   | resnet50_v1.5 pruned0.5      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 84   | resnet50_v1.5 pruned0.6      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 85   | resnet50_v1.5 pruned0.7      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 86   | OFA-resnet50                 | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 87   | OFA-resnet50 pruned0.45      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 88   | OFA-resnet50 pruned0.60      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 89   | OFA-resnet50 pruned0.74      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 90   | OFA-resnet50 pruned0.88      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 91   | OFA-depthwise-resnet50       | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 92   | vehicle type classification  | PyTorch    |       √       |       √       |   ×    |                                |
| 93   | vehicle make classification  | PyTorch    |       √       |       √       |   ×    |                                |
| 94   | vehicle color classification | PyTorch    |       √       |       √       |   ×    |                                |
| 95   | OFA-yolo                     | PyTorch    |       √       |       √       |   ×    |                                |
| 96   | OFA-yolo pruned0.3           | PyTorch    |       √       |       √       |   √    |                                |
| 97   | OFA-yolo pruned0.6           | PyTorch    |       √       |       √       |   √    |                                |
| 98   | yolox-nano                   | PyTorch    |       √       |       √       |   ×    |                                |
| 99   | yolov4csp                    | PyTorch    |       √       |       √       |   ×    |                                |
| 100  | yolov5-large                 | PyTorch    |       √       |       √       |   ×    |                                |
| 101  | yolov5-nano                  | PyTorch    |       √       |       √       |   ×    |                                |
| 102  | yolov5s6                     | PyTorch    |       √       |       √       |   ×    |                                |
| 103  | yolov6m                      | PyTorch    |       √       |       √       |   ×    |                                |
| 104  | pointpillars                 | PyTorch    |       √       |       √       |   ×    |                                |
| 105  | CLOCs                        | PyTorch    |       √       |       √       |   ×    |                                |
| 106  | Enet                         | PyTorch    |       √       |       √       |   ×    |                                |
| 107  | SemanticFPN-resnet18         | PyTorch    |       √       |       √       |   ×    |                                |
| 108  | SemanticFPN-mobilenetv2      | PyTorch    |       √       |       √       |   ×    |                                |
| 109  | salsanext pruned0.60         | PyTorch    |       √       |       √       |   √    |                                |
| 110  | salsanextv2 pruned0.75       | PyTorch    |       √       |       √       |   √    |                                |
| 111  | SOLO                         | PyTorch    |       √       |       √       |   ×    |                                |
| 112  | HRNet                        | PyTorch    |       √       |       √       |   ×    |                                |
| 113  | CFLOW                        | PyTorch    |       √       |       √       |   ×    |                                |
| 114  | 3D-UNET                      | PyTorch    |       √       |       √       |   ×    |                                |
| 115  | MaskRCNN                     | PyTorch    |       √       |       √       |   ×    |                                |
| 116  | bert-base                    | PyTorch    |       √       |       √       |   ×    |                                |
| 117  | bert-large                   | PyTorch    |       √       |       √       |   ×    |                                |
| 118  | bert-tiny                    | PyTorch    |       √       |       √       |   ×    |                                |
| 119  | face-mask-detection          | PyTorch    |       √       |       √       |   ×    |                                |
| 120  | movenet                      | PyTorch    |       √       |       √       |   ×    |                                |
| 121  | fadnet                       | PyTorch    |       √       |       √       |   ×    |                                |
| 122  | fadnet pruned0.65            | PyTorch    |       √       |       √       |   √    |                                |
| 123  | fadnetv2                     | PyTorch    |       √       |       √       |   ×    |                                |
| 124  | fadnetv2 pruned0.51          | PyTorch    |       √       |       √       |   √    |                                |
| 125  | psmnet pruned0.68            | PyTorch    |       √       |       √       |   √    |                                |
| 126  | pmg                          | PyTorch    |       √       |       √       |   ×    |                                |
| 127  | SESR-S                       | PyTorch    |       √       |       √       |   ×    |                                |
| 128  | OFA-rcan                     | PyTorch    |       √       |       √       |   ×    |                                |
| 129  | DRUNet                       | PyTorch    |       √       |       √       |   ×    |                                |
| 130  | xilinxSR                     | PyTorch    |       √       |       √       |   ×    |                                |

</details>

**UIF 1.1 Models for MIGraphX**
	
<details>
 <summary>Click here to view details</summary>	

| #    | Model                   | Original Platform | Converted Format | Datatype FP32 | Datatype FP16 | Datatype INT8 | Pruned | Reminder for limited use scope |
| ---- | ----------------------- | ----------------- | ---------------- | ------------- | ------------- | ------------- | ------ | ------------------------------ |
| 1    | Resnet50_v1             | TensorFlow        | .PB              | √             | √             | ×             | ×      | Non-Commercial Use Only        |
| 2    | Resnet50_v1 pruned0.5   | TensorFlow        | .PB              | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 3    | Resnet50_v1 pruned0.7   | TensorFlow        | .PB              | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 4    | Inception_v3            | TensorFlow        | .PB              | √             | √             | ×             | ×      | Non-Commercial Use Only        |
| 5    | Inception_v3 pruned0.4  | TensorFlow        | .PB              | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 6    | Inception_v3 pruned0.6  | TensorFlow        | .PB              | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 7    | Mobilenet_v1            | TensorFlow        | .PB              | √             | √             | ×             | ×      | Non-Commercial Use Only        |
| 8    | Mobilenet_v1 pruned0.3  | TensorFlow        | .PB              | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 9    | Mobilenet_v1 pruned0.5  | TensorFlow        | .PB              | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 10   | Resnet34-ssd            | TensorFlow        | .PB              | √             | √             | ×             | ×      | Non-Commercial Use Only        |
| 11   | Resnet34-ssd pruned0.19 | TensorFlow        | .PB              | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 12   | Resnet34-ssd pruned0.29 | TensorFlow        | .PB              | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 13   | Bert-base               | PyTorch           | .ONNX            | √             | √             | ×             | ×      |                                |
| 14   | Bert-large              | PyTorch           | .ONNX            | √             | √             | ×             | ×      |                                |
| 15   | DLRM                    | PyTorch           | .ONNX            | √             | √             | ×             | ×      | Non-Commercial Use Only        |
| 16   | Resnet50_v1.5           | PyTorch           | .ONNX            | √             | √             | ×             | ×      | Non-Commercial Use Only        |
| 17   | Resnet50_v1.5 pruned0.4 | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 18   | Resnet50_v1.5 pruned0.6 | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 19   | Inception_v3            | PyTorch           | .ONNX            | √             | √             | ×             | ×      | Non-Commercial Use Only        |
| 20   | Inception_v3 pruned0.4  | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 21   | Inception_v3 pruned0.6  | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 22   | Reid_resnet50           | PyTorch           | .ONNX            | √             | √             | ×             | ×      | Non-Commercial Use Only        |
| 23   | Reid_resnet50 pruned0.6 | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 24   | Reid_resnet50 pruned0.7 | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 25   | OFA_resnet50 pruned0.45 | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 26   | OFA_resnet50 pruned0.60 | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 27   | OFA_resnet50 pruned0.74 | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 28   | OFA_resnet50 pruned0.88 | PyTorch           | .ONNX            | √             | √             | ×             | √      | Non-Commercial Use Only        |
| 29   | GPT-2 small             | PyTorch           | .ONNX            | √             | √             | ×             | ×      |                                |
| 30   | DistillGPT              | PyTorch           | .ONNX            | √             | √             | ×             | ×      |                                |

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

Model name: `F_M_(D)_H_W_(P)_C_V_Z`
* `F` specifies the training framework: `tf` is TensorFlow 1.x, `tf2` is TensorFlow 2.x, `pt` is PyTorch.
* `M` specifies the model.
* `D` specifies the dataset. It is optional depending on whether the dataset is public or private.
* `H` specifies the height of input data.
* `W` specifies the width of input data.
* `P` specifies the pruning ratio, meaning how much computation is reduced. It is optional depending on whether the model is pruned or not.
* `C` specifies the computation of the model: how many Gops per image.
* `V` specifies the version of UIF.
* `Z` specifies the version of ZenDNN or MIGraphX.

For example, `pt_inceptionv3_imagenet_299_299_0.6_4.25G_1.1_Z4.0` is an `Inception v3` model trained with `PyTorch` using `Imagenet` dataset, the input size is `299*299`, `60%` pruned, the computation per image is `4.25G flops`, the UIF release version is `1.1`, and ZenDNN version is `4.0`.

`pt_resnet50v1.5_imagenet_224_224_8.2G_1.1_M2.4` is a `resnet50 v1.5` model trained with `PyTorch` using the `Imagenet` dataset, the input size is `224*224`, `No` pruned, the computation per image is `8.2G flops`, the UIF release version is `1.1`, and the MIGraphX version is `2.4`.

### 2.1.2: Model List

Visit [model-list](/docs/2_model_setup/model-list). The models are named according to standard naming rules. From here, you can get a download link and MD5 checksum of all released models running on different hardware platforms. You can download it manually or use the automatic download script described below to search for models by keyword.                                 


# 2.2: Get ZenDNN Models from UIF Model Zoo

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
    tf:tensorflow1.x  tf2:tensorflow2.x  cf:caffe  dk:darknet  pt:pytorch  all: list all model
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
	1 : pt_resnet50_imagenet_224_224_8.2G_1.1_Z4.0
	2 : pt_resnet50_imagenet_224_224_0.3_5.8G_1.1_Z4.0
	3 : pt_resnet50_imagenet_224_224_0.4_4.9G_1.1_Z4.0
	4 : pt_resnet50_imagenet_224_224_0.5_4.1G_1.1_Z4.0
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
    pt_resnet50_imagenet_224_224_0.4_4.9G_1.1_Z4.0.zip
	                                              100.0%|100%
	done
    ```
    
# 2.3: Get MIGraphX Models from UIF Model Zoo

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
    tf:tensorflow1.x  tf2:tensorflow2.x  cf:caffe  dk:darknet  pt:pytorch  all: list all model
    input:
    ```
2. Download UIF 1.1 GPU models. Provide `pt` as input to get the list of models.

    ```
	input:pt
	chose model
	0 : all
	1 : pt_resnet50v1.5_imagenet_224_224_8.2G_1.1_M2.4.zip
	2 : pt_resnet50v1.5_imagenet_224_224_8.2G_1.1_M2.4_MI100.zip
	3 : pt_resnet50v1.5_imagenet_224_224_8.2G_1.1_M2.4_MI210.zip

	...
	input num:
    ```

The models with 1.1 as suffix are UIF 1.1 models. MI100 means the model has been tuned for MI-100 GPU, and MI210 indicates the model has been tuned for MI-210 GPU.  Without either of these suffixes, the model is a model version that should be used for training, including the GPU example later in this chapter.

    ```
	input num:1
	chose model type
	0: all
	1: GPU
	2: MI100
	3: MI210
	...
	...
    input num:
    
    ```
    
Provide `1` as input to download the GPU model.

    ```
    input num:1
    pt_resnet50v1.5_imagenet_224_224_8.2G_1.1_M2.4.zip
	                                              100.0%|100%
	done
    ```

# 2.4: Set Up MIGraphX YModel

YModel is designed to provide significant inference performance through MIGraphX. Prior to the introduction of YModel, performance tuning was conditioned by having tuned kernel configs stored in a `/home` local User DB. If users move their model to a different server or allow a different user to use it, they would have to run through the MIOpen tuning process again to populate the next User DB with the best kernel configs and corresponding solvers. Tuning is time consuming, and if the users have not performed tuning, they would see discrepancies between expected or claimed inference performance and the actual inference performance. This leads to repetitive and time-consuming tuning tasks for each user. 

MIGraphX introduces a feature known as YModel that stores the kernel config parameters found during tuning in the same file as the model itself. This ensures the same level of expected performance even when a model is copied to a different user or system.   

**Note:** The YModel feature is available in the ROCm™ v5.4.1 and UIF v1.1 release. For more information on ROCm, refer to https://docs.amd.com.

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

# 2.5: Get Vitis AI Models from UIF Model Zoo

Follow the instructions in the [Vitis AI](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo) Model Zoo page.

<hr/>

[< Previous](/docs/1_installation/installation.md) | [Next >](/docs/2_model_setup/gpu_model_example.md)

<hr/>

 # License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.

