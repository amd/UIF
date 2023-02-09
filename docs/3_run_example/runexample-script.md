<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.1 User Guide </h1>
    </td>
 </tr>
 <tr>
 <td align="center"><h1> Step 3.1: Run a CPU Example</h1>
 </td>
 </tr>
</table>


# Table of Contents
- [3.1.1: Sample Run with TensorFlow+ZenDNN](#311-sample-run-with-tensorflowzendnn)
  - [3.1.1.1: Preparation](#3111-preparation)
  - [3.1.1.2: Run Sample](#3112-run-sample)
  - [3.1.1.3: Parameter Descriptions](#3113-parameter-descriptions)
  - [3.1.1.4: Example](#3114-example)

  _Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._

# 3.1.1: Sample Run with TensorFlow+ZenDNN

This repo includes a sample to run a demo application. The demo loads images from a given folder, infers on the images with the given TensorFlow model, and prints out Top1 and Top5 classes.

## 3.1.1.1: Preparation

1. Set up the environment.

    Install TensorFlow+ZenDNN. For more information, see the [Installation](/docs/1_installation/installation.md#131-tensorflowzendnn) section. Pillow is required, and you can install it with the following command.

    ```
    pip install pillow
    ```

2. Download the models.

    Download the TensorFlow Image Recognition (ResNet50, InceptionV3, MobileNetv1, VGG16) model from the model zoo. For more information, see the [Model Setup](/docs/2_model_setup/uifmodelsetup.md) section.

3. Get data.

    Due to copyright reasons, images are not included with this sample.

    To use the sample, create a directory and copy the required images into the folder. The images in this folder are used for inference.

## 3.1.1.2: Run Sample

To run the sample, change the directory into the `samples/zendnn` directory. Use the following command with the correct parameters:

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

### 3.1.1.3: Parameter Descriptions
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

### 3.1.1.4: Example

This tutorial uses ResNet50 as an example. To run for a single image:

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

<hr/>

[< Previous](/docs/2_model_setup/gpu_model_example.md) | [Next >](/docs/3_run_example/inference_server_example.md)

<hr/>

 # License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.


