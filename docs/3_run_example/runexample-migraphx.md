<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.1 User Guide </h1>
    </td>
 </tr>
 <tr>
 <td align="center"><h1>Step 3.3: Run a MIGraphX Example</h1>
 </td>
 </tr>
</table>

# Table of Contents
- [3.3.1: Sample Run with MIGraphX](#331-sample-run-with-migraphx)
  - [3.3.1.1: Preparation](#3311-preparation)
  - [3.3.1.2: Run Sample](#3312-run-sample)
  - [3.3.1.3: Parameter Descriptions](#3313-parameter-descriptions)
  - [3.3.1.4: Example](#3314-example)

  _Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._

# 3.3.1: Sample Run with MIGraphX

This repo also contains a sample to run a demo application with MIGraphX.  The demo loads a single image, infers on the image using an Imagenet trained model, and prints out the Top1 class.

## 3.3.1.1: Preparation


1. Run the following command to install OpenCV used by the sample:

```
pip install opencv-python
```


## 3.3.1.2: Run Sample

2. To run the sample, change the directory to the `samples/migraphx` directory.  Use the following command with the correct parameters:


```
python migx_sample.py \
    --onnx_file <onnx_file> \
    --image <image_file>
```
            

## 3.3.1.3: Parameter Descriptions

```
                              --onnx_file:         name of an imagenet ONNX model file
                              --mxr_file:           name of an MIGraphX YModel file
                              --image:               name of input image file
```

**Note:** Either the `--onnx_file` or `--mxr_file` options should be given.

## 3.3.1.4: Example

```
python migx_sample.py --onnx resnet50_fp32.onnx --image cow.jpg
```
    
             
<hr/>

[< Previous](/docs/3_run_example/inference_server_example.md) | [Next >](/docs/4_deploy_your_own_model/prune_model/prunemodel.md)

<hr/>

 # License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.


