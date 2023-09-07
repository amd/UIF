<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.2 User Guide </h1>
    </td>
 </tr>
 <tr>
 <td align="center"><h1>Step 4.2: Quantize Model with UIF Quantizer for Target Platforms (GPU/CPU/FPGA)</h1>
 </td>
 </tr>
</table>

# Table of Contents
- [4.2.1: Quantize PyTorch Models](#421-quantize-pytorch-models)
  - [4.2.1.1: Vitis AI PyTorch Quantizer](#4211-vitis-ai-pytorch-quantizer)
  - [4.2.1.2: Configuration of Quantization Strategy for UIF](#4212-configuration-of-quantization-strategy-for-uif)
- [4.2.2: Quantize TensorFlow Models](#422-quantize-tensorflow-models)
  - [4.2.2.1: Installation](#4221-installation)
  - [4.2.2.2: Running vai_q_tensorflow](#4222-running-vai_q_tensorflow)
  - [4.2.2.3: vai_q_tensorflow Quantization Aware Training](#4223-vai_q_tensorflow-quantization-aware-training)
  - [4.2.2.4: vai_q_tensorflow Supported Operations and APIs](#4224-vai_q_tensorflow-supported-operations-and-apis)
  - [4.2.2.5: vai_q_tensorflow Usage](#4225-vai_q_tensorflow-usage)
- [4.2.3: Quantize ONNX Models](#423-quantize-onnx-models)
  - [4.2.3.1 Test Environment](#4231-test-environment)
  - [4.2.3.2 Installation](#4232-installation)
  - [4.2.3.3 Post Training Quantization](#4233-post-training-quantizationptq---static-quantization)
  - [4.2.3.4 Running vai_q_onnx](#4234-running-vai_q_onnx)
  - [4.2.3.5 List of Vai_q_onnx Supported Quantized Ops](#4235-list-of-vai_q_onnx-supported-quantized-ops)
  - [4.2.3.6 vai_q_onnx APIs](#4236-vai_q_onnx-apis)

  _Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._

# 4.2.1: Quantize PyTorch Models

## 4.2.1.1: Vitis AI PyTorch Quantizer

To quantize UIF models, use the Vitis&trade; AI PyTorch Quantizer tool (vai_q_pytorch). Refer to the [vai_q_pytorch](https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_quantizer/vai_q_pytorch) and the relevant [user guide](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/PyTorch-Version-vai_q_pytorch) for detailed information.

## 4.2.1.2: Configuration of Quantization Strategy for UIF

- [pt_quant_config.json](pt_quant_config.json) is a special configuration for ZenDNN deployment. All quantization for ZenDNN deployment must apply this configuration.
- For multiple quantization strategy configurations, vai_q_pytorch supports quantization configuration file in JSON format. You only need to pass the configuration file to the torch_quantizer API.
- For detailed information of the JSON file contents, refer to [Quant_Config.md](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/doc/Quant_Config.md).

## 4.2.1.3: Quick Start in UIF Docker Environment

Using [pt_resnet18_quant.py](pt_resnet18_quant.py) as an example:

1. Copy [pt_resnet18_quant.py](pt_resnet18_quant.py) to the Docker® environment.
2. Download the pre-trained [Resnet18 model](https://download.pytorch.org/models/resnet18-5c106cde.pth): 
    ```shell
    wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O resnet18.pth
    ```
3. Prepare Imagenet validation images. See [PyTorch example repo](https://github.com/pytorch/examples/tree/master/imagenet) for reference.
4. Modify default data_dir and model_dir in [pt_resnet18_quant.py](pt_resnet18_quant.py).
5. Quantize, using a subset (200 images) of validation data for calibration. Because we are in quantize calibration process, the displayed loss and accuracy are meaningless.

  ```shell
  python pt_resnet18_quant.py --quant_mode calib --subset_len 200 --config_file ./pt_quant_config.json
  ```
6. Evaluate the quantized model. 

  ```shell
  python pt_resnet18_quant.py --quant_mode test
  ```

### Write Quantization Code Based on Float Model

1. Import vai_q_pytorch modules. <br>
   ```py
    from pytorch_nndct.apis import torch_quantizer
   ```
2. Generate a quantizer with quantization needed input and get converted model. <br>
   ```py
    input = torch.randn([batch_size, 3, 224, 224])
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file)
    quant_model = quantizer.quant_model
   ```
3. Forwarding with converted model. <br>
   ```py
    acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)
   ```
4. Output quantization result and deploy model. <br>
   ```py
   if quant_mode == 'calib':
     quantizer.export_quant_config()
   if quant_mode == 'test':
     quantizer.export_onnx_model()
   ```

### Run and Output Results

Before running commands, introducing the log message in vai_q_pytorch. vai_q_pytorch log messages have special colors and the keyword prefix "VAIQ_*".  vai_q_pytorch log message types include error, warning, and note. 
Pay attention to vai_q_pytorch log messages to check the flow status.<br>

* Run command with `--quant_mode calib` to the quantize model.

```py
    python pt_resnet18_quant.py --quant_mode calib --subset_len 200 --config_file ./pt_quant_config.json
```
When doing calibration forward, the float evaluation flow is borrowed to minimize code changes from the float script, which means that loss and accuracy information is displayed at the end. Because these loss and accuracy values are meaningless at this point in the process, they can be skipped. Attention should be given to the colorful log messages with the special keywords prefix VAIQ_*.

It is important to control iteration numbers during quantization and evaluation. Generally, 100-1000 images are enough for quantization, and the whole validation set is required for evaluation. The iteration numbers can be controlled in the data loading part.
In this case, the argument `subset_len` controls how many images used for network forwarding. However, if the float evaluation script does not have an argument with similar role, it is better to add one, otherwise it should be changed manually.

If this quantization command runs successfully, two important files are generated under output directory `./quantize_result`.

```
    ResNet.py: converted vai_q_pytorch format model, 
    Quant_info.json: quantization steps of tensors got. (Keep it for evaluation of quantized model)
```
To evaluate the quantized model, run the following command:

```shell
    python pt_resnet18_quant.py --quant_mode test 
```
When this command finishes, the displayed accuracy is the right accuracy for quantized model. <br> 

### Fast Finetune Model 

Sometimes direct quantization accuracy is not high enough, and finetuning of model parameters is necessary to recover accuracy:<br>

- Fast finetuning is not real training of the model and only needs a limited number of iterations. For classification models on the Imagenet dataset, 5120 images are enough in general.
- It only requires some modification based on the evaluation model script, and does not require you to set up Optimizer for training.
- A function for model forwarding iteration is needed and is called as part of fast finetuning. 
- Re-calibration with original inference code is highly recommended.
- Example code in [pt_resnet18_quant.py](pt_resnet18_quant.py) is as follows:

```py
# fast finetune model or load finetuned parameter before test 
  if finetune == True:
      ft_loader, _ = load_data(
          subset_len=5120,
          train=False,
          batch_size=batch_size,
          sample_method='random',
          data_dir=args.data_dir,
          model_name=model_name)
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
```
- For [pt_resnet18_quant.py](pt_resnet18_quant.py), use the command line to perform parameter fast finetuning and re-calibration:

  ```shell
  python pt_resnet18_quant.py --quant_mode calib --fast_finetune --config_file ./pt_quant_config.json
  
  ```
- Use the command line to test fast finetuned quantized model accuracy as follows:

  ```shell
  python pt_resnet18_quant.py --quant_mode test --fast_finetune

  ```

### Finetune Quantized Model

- This mode can be used to finetune a quantized model (loading float model parameters), as well as to do quantization-aware-training (QAT) from scratch.
- It is necessary to add some vai_q_pytorch interface functions based on the float model training script.
- The mode requires that the trained model cannot use the +/- operator in the model forwarding code. Replace them with the torch.add/torch.sub module.
- For detailed information, refer to the [Vitis AI User Guide](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/vai_q_pytorch-QAT).

# 4.2.2: Quantize TensorFlow Models 

vai_q_tensorflow is a Vitis AI quantizer for TensorFlow. It supports FPGA-friendly quantization for TensorFlow models. After quantization, models can be deployed to FPGA devices. vai_q_tensorflow is a component of [Vitis AI](https://github.com/Xilinx/Vitis-AI), a development stack for AI inference on AMD hardware platforms.

**Note:** You can download the AMD prebuilt version in [Vitis AI](https://www.xilinx.com/products/design-tools/vitis.html). See the [Vitis AI User Guide](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai) for further details.

##  4.2.2.1: Installation

Tested environment:

* Ubuntu 16.04
* GCC 4.8
* Bazel 0.24.1
* Python 3.6
* CUDA 10.1 + CUDNN 7.6.5

Prerequisites: 

*  Install [Bazel 0.24.1](https://bazel.build/).
*  (GPU version) Install CUDA and CUDNN.
*  Install Python prerequisites:
```shell
$ pip install -r requirements.txt 
```

Option 1. Build wheel package and install:

```shell
# CPU-only version
$ ./configure # Input "N" when asked "Do you wish to build TensorFlow with CUDA support?". For other questions, use default value or set as you wish.
$ sh rebuild_cpu.sh
# GPU version
$ ./configure # Input "Y" when asked "Do you wish to build TensorFlow with CUDA support?". For other questions, use default value or set as you wish.
$ sh rebuild_gpu.sh
```

Option 2. Build conda package (you need [anaconda](https://www.anaconda.com/) for this option):

```shell
# CPU-only version
$ conda build --python=3.6 vai_q_tensorflow_cpu_feedstock --output-folder ./conda_pkg/
# GPU version
$ conda build --python=3.6 vai_q_tensorflow_gpu_feedstock --output-folder ./conda_pkg/
# Install conda package
$ conda install --use-local ./conda_pkg/linux-64/vai_q_tensorflow-1.0-py36h605774d_1.tar.bz2
```

Validate installation:

```shell
$ vai_q_tensorflow --version
$ vai_q_tensorflow --help
```

**Note:** This tool is based on TensorFlow 1.15. For more information on build and installation, refer to the [TensorFlow Installation Guide](https://www.tensorflow.org/install/source), specifically [Docker linux build](https://www.tensorflow.org/install/source#docker_linux_builds) or [Windows installation](https://www.tensorflow.org/install/source_windows).

## 4.2.2.2: Running vai_q_tensorflow

### Preparing the Float Model and Related Input Files

Before running vai_q_tensorflow, prepare the frozen inference TensorFlow model in floating-point format and calibration set, including the files listed in the following table.

**Input Files for vai_q_tensorflow**

|No.|Name|Description|
| :--- | :--- | :--- |
|1|frozen_graph.pb|Floating-point frozen inference graph. Ensure that the graph is the inference graph rather than the training graph.|
|2|calibration dataset|A subset of the training dataset containing 100 to 1000 images.|
|3|input_fn|An input function to convert the calibration dataset to the input data of the frozen_graph during quantize calibration. Usually performs data pre-processing and augmentation.|

#### **Generating the Frozen Inference Graph**

Training a model with TensorFlow 1.x creates a folder containing a GraphDef file (usually ending with `a.pb` or `.pbtxt` extension) and a set of checkpoint files. What you need for mobile or embedded deployment is a single GraphDef file that has been frozen, or had its variables converted into inline constants, so everything is in one file. To handle the conversion, TensorFlow provides `freeze_graph.py`, which is automatically installed with the vai_q_tensorflow quantizer.

An example of command-line usage is as follows:

```shell
$ freeze_graph \
--input_graph /tmp/inception_v1_inf_graph.pb \
--input_checkpoint /tmp/checkpoints/model.ckpt-1000 \
--input_binary true \
--output_graph /tmp/frozen_graph.pb \
--output_node_names InceptionV1/Predictions/Reshape_1
```
The `–input_graph` should be an inference graph other than the training graph. Because the operations of data preprocessing and loss functions are not needed for inference and deployment, the `frozen_graph.pb` should only include the main part of the model. In particular, the data preprocessing operations should be taken in the `input_fn` to generate correct input data for quantize calibration.

**Note:** Some operations, such as dropout and batchnorm, behave differently in the training and inference phases. Ensure that they are in the inference phase when freezing the graph. For examples, you can set the flag `is_training=false` when using `tf.layers.dropout/tf.layers.batch_normalization`. For models using `tf.keras`, call `tf.keras.backend.set_learning_phase(0)` before building the graph.

**Tip:** Type `freeze_graph --help` for more options.

The input and output node names vary depending on the model, but you can inspect and estimate them with the vai_q_tensorflow quantizer. See the following code snippet for an example:

```shell
$ vai_q_tensorflow inspect --input_frozen_graph=/tmp/inception_v1_inf_graph.pb
```

The estimated input and output nodes cannot be used for quantization if the graph has in-graph pre- and post-processing. This is because some operations are not quantizable and can cause errors when compiled by the Vitis AI compiler, if you deploy the quantized model to the DPU.

Another way to get the input and output name of the graph is by visualizing the graph. Both TensorBoard and Netron can do this. See the following example, which uses Netron:

```shell
$ pip install netron
$ netron /tmp/inception_v3_inf_graph.pb
```

#### **Preparing the Calibration Dataset and Input Function**

The calibration set is usually a subset of the training/validation dataset or actual application images (at least 100 images for performance). The input function is a Python importable function to load the calibration dataset and perform data preprocessing. The vai_q_tensorflow quantizer can accept an input_fn to do the preprocessing, which is not saved in the graph. If the preprocessing subgraph is saved into the frozen graph, the input_fn only needs to read the images from dataset and return a feed_dict.

The format of input function is `module_name.input_fn_name`, (for example,
`my_input_fn.calib_input`). The `input_fn` takes an int object as input, indicating the calibration step number, and returns a dict ("`placeholder_name, numpy.Array`") object for each call, which is fed into the placeholder nodes of the model when running inference. The `placeholder_name` is always the input node of the frozen graph, that is to say, the node receiving input data. The `input_nodes`, in the vai_q_tensorflow options, indicate where quantization starts in the frozen graph. The `placeholder_names` and the `input_nodes` options are sometimes different. For example, when the frozen graph includes in-graph preprocessing, the placeholder_name is the input of the graph, although it is recommended that `input_nodes` be set to the last node of preprocessing. The shape of `numpy.array` must be consistent with the placeholders. See the following pseudo code example:

```python
$ “my_input_fn.py”
def calib_input(iter):
"""
A function that provides input data for the calibration
Args:
iter: A `int` object, indicating the calibration step number
Returns:
dict( placeholder_name, numpy.array): a `dict` object, which will be fed into the model
"""
image = load_image(iter)
preprocessed_image = do_preprocess(image)
return {"placeholder_name": preprocessed_images}
```

### Quantizing the Model Using vai_q_tensorflow

Run the following commands to quantize the model:

```shell
$vai_q_tensorflow quantize \
--input_frozen_graph frozen_graph.pb \
--input_nodes ${input_nodes} \
--input_shapes ${input_shapes} \
--output_nodes ${output_nodes} \
--input_fn input_fn \
[options]
```
The `input_nodes` and `output_nodes` arguments are the name list of input nodes of the quantize graph. They are the start and end points of quantization. The main graph between them is quantized if it is quantizable, as shown in the following figure.

<div align="center">
  <img src="Quantization_Flow_for_TensorFlow.jpg">
</div>

It is recommended to set `–input_nodes` to be the last nodes of the preprocessing part. Similarly, it is advisable to set *-output_nodes* to be the last nodes of the main graph part because some operations in the pre- and postprocessing parts are not quantizable and might cause errors when compiled by the Vitis AI quantizer if you need to deploy the quantized model to the DPU.

The input nodes might not be the same as the placeholder nodes of the graph. If no in-graph preprocessing part is present in the frozen graph, the placeholder nodes should be set to input nodes.

The `input_fn` should be consistent with the placeholder nodes.

`[options]` stands for optional parameters. The most commonly used options are as follows:

- **weight_bit**: Bit width for quantized weight and bias (default is 8).

- **activation_bit**: Bit width for quantized activation (default is 8).

- **method**: Quantization methods, including 0 for non-overflow, 1 for min-diffs, and 2 for mindiffs with normalization. The non-overflow method ensures that no values are saturated.

### Generating the Quantized Model

After the successful execution of the `vai_q_tensorflow` command, one output file is generated in the `${output_dir}` location `quantize_eval_model.pb`. This file is used to evaluate the CPU/GPUs, and can be used to simulate the results on hardware. Run import `tensorflow.contrib.decent_q` explicitly to register the custom quantize operation because `tensorflow.contrib` is now lazy-loaded.

**vai_q_tensorflow Output Files**

|No.|Name|Description|
| :--- | :--- | :--- |
|1|deploy_model.pb|Quantized model for the Vitis AI compiler (extended TensorFlow format) for targeting DPUCZDX8G implementations.|
|2|quantize_eval_model.pb|Quantized model for evaluation (also, the Vitis AI compiler input for most DPU architectures, like DPUCAHX8H and DPUCADF8H).|

### Evaluating the Quantized Model (Optional)

If you have scripts to evaluate floating point models, like the models in [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo), apply the following two changes to evaluate the quantized model:

- Prepend the float evaluation script with `from tensorflow.contrib import decent_q` to register the quantize operation.

- Replace the float model path in the scripts to quantization output model `quantize_results/quantize_eval_model.pb`.

- Run the modified script to evaluate the quantized model.

### Dumping the Simulation Results (Optional)

vai_q_tensorflow dumps the simulation results with the `quantize_eval_model.pb` generated by the quantizer. This allows you to compare the simulation results on the CPU/GPU with the output values on the DPU.

To dump the quantize simulation results, run the following commands:

```shell
$vai_q_tensorflow dump \
--input_frozen_graph quantize_results/quantize_eval_model.pb \
--input_fn dump_input_fn \
--max_dump_batches 1 \
--dump_float 0 \
--output_dir quantize_results
```
The input_fn for dumping is similar to the input_fn for quantize calibration, but the batch size is often set to 1 to be consistent with the DPU results.

If the command executes successfully, dump results are generated in `_${output_dir}_`. There are folders in `${output_dir}`, and each folder contains the dump results for a batch of input data. Results for each node are saved separately. For each quantized node, results are saved in `_int8.bin` and `_int8.txt` format. If `dump_float` is set to 1, the results for unquantized nodes are dumped. The / symbol is replaced by _ for simplicity. Examples for dump results are shown in the following table.

**Examples for Dump Results**

|Batch No.|Quant|Node Name|Saved files|
| :--- | :--- | :--- | :--- |
|1|Yes|resnet_v1_50/conv1/biases/wquant|{output_dir}/dump_results_1/resnet_v1_50_conv1_biases_wquant_int8.bin  {output_dir}/dump_results_1/resnet_v1_50_conv1_biases_wquant_int8.txt|
|2|No|resnet_v1_50/conv1/biases|{output_dir}/dump_results_2/resnet_v1_50_conv1_biases.bin  {output_dir}/dump_results_2/resnet_v1_50_conv1_biases.txt|

## 4.2.2.3: vai_q_tensorflow Quantization Aware Training

Quantization aware training (QAT) is similar to float model training/finetuning, but in QAT, the vai_q_tensorflow APIs are used to rewrite the float graph to convert it to a quantized graph before the training starts. The typical workflow is as follows:

1. **Preparation:** Before QAT, prepare the following files:

**Input Files for vai_q_tensorflow QAT**

|No.|Name|Description|
| :--- | :--- | :--- |
|1|Checkpoint files|Floating-point checkpoint files to start from. Omit this if you are training the model from scratch.|
|2|Dataset|The training dataset with labels.|
|3|Train Scripts|The Python scripts to run float train/finetuning of the model.|

2. **Evaluate the float model (optional):** Evaluate the float checkpoint files first before doing quantize finetuning to check the correctness of the scripts and dataset. The accuracy and loss values of the float checkpoint can also be a baseline for QAT.

3. **Modify the training scripts:** To create the quantize training graph, modify the training scripts to call the function after the float graph is built. The following is an example:

```python
# train.py

# ...

# Create the float training graph
model = model_fn(is_training=True)

# *Set the quantize configurations
from tensorflow.contrib import decent_q
q_config = decent_q.QuantizeConfig(input_nodes=['net_in'], output_nodes=['net_out'],input_shapes=[[-1, 224, 224, 3]])

# *Call Vai_q_tensorflow api to create the quantize training graph
decent_q.CreateQuantizeTrainingGraph(config=q_config)

# Create the optimizer
optimizer = tf.train.GradientDescentOptimizer()

# start the training/finetuning, you can use sess.run(), tf.train, tf.estimator, tf.slim and so on
# ...
```

The `QuantizeConfig` contains the configurations for quantization.

Some basic configurations such as `input_nodes`, `output_nodes`, and `input_shapes` need to be set according to your model structure. 

Other configurations such as `weight_bit`, `activation_bit`, and `method` have default values and can be modified as needed. See [vai_q_tensorflow Usage](#4225-vai_q_tensorflow-usage) for detailed information of all the configurations.

- `input_nodes`/`output_nodes`: They are used together to determine the subgraph range you want to quantize. The pre-processing and post-processing components are usually not quantizable and should be out of this range. The `input_nodes` and `output_nodes` should be the same for the float training graph and the float evaluation graph to match the quantization operations between them.

    **Note:** Operations with multiple output tensors (such as FIFO) are currently not supported. You can add a `tf.identity` node to make an alias for the `input_tensor` to make a single output input node.

- input_shapes: The shape list of input_nodes must be a 4-dimension shape for each node. The information is comma separated, for example, [[1,224,224,3] [1, 128, 128, 1]]; support unknown size for batch_size, for example, [[-1,224,224,3]].

4. **Evaluate the quantized model and generate the frozen model:** After QAT, generate the frozen model after evaluating the quantized graph with a checkpoint file. This can be done by calling the following function after building the float evaluation graph. As the freeze process depends on the quantize evaluation graph, they are often called together.

  **Note**: Function `decent_q.CreateQuantizeTrainingGraph` and `decent_q.CreateQuantizeEvaluationGraph` modify the default graph in TensorFlow. They need to be called on different graph phases. `decent_q.CreateQuantizeTrainingGraph` needs to be called on the float training graph while `decent_q.CreateQuantizeEvaluationGraph` needs to be called on the float evaluation graph. `decent_q.CreateQuantizeEvaluationGraph` cannot be called right after calling function `decent_q.CreateQuantizeTrainingGraph`, because the default graph has been converted to a quantize training graph. The correct way is to call it right after the float model creation function.

```python
# eval.py

# ...

# Create the float evaluation graph
model = model_fn(is_training=False)

# *Set the quantize configurations
from tensorflow.contrib import decent_q
q_config = decent_q.QuantizeConfig(input_nodes=['net_in'], output_nodes=['net_out'], input_shapes=[[-1, 224, 224, 3]])

# *Call Vai_q_tensorflow api to create the quantize evaluation graph
decent_q.CreateQuantizeEvaluationGraph(config=q_config)

# *Call Vai_q_tensorflow api to freeze the model and generate the deploy model
decent_q.CreateQuantizeDeployGraph(checkpoint="path to checkpoint folder", config=q_config)

# start the evaluation, users can use sess.run, tf.train, tf.estimator, tf.slim and so on
# ...
```

### Generated Files

After you have performed the previous steps, the following files are generated in the *${output_dir}* location:

**Generated File Information**

|Name|TensorFlow Compatible|Usage|Description|
| :--- | :--- | :--- | :--- |
|quantize_train_graph.pb|Yes|Train|The quantize train graph.|
|quantize_eval_graph_{suffix}.pb|Yes|Evaluation with checkpoint|The quantize evaluation graph with quantize information frozen inside. There are weights in this file and it should be used together with the checkpoint file in evaluation.|
|quantize_eval_model_{suffix}.pb|Yes|1: Evaluation  <br>  2: Dump  <br>  3: Input to VAI compiler (DPUCAHX8H)|The frozen quantize evaluation graph, weights in the checkpoint, and quantize information are frozen inside. It can be used to evaluate the quantized model on the host or to dump the outputs of each layer for cross check with DPU outputs. The XIR compiler uses it as an input.|

The suffix contains the iteration information from the checkpoint file and the date information. For example, if the checkpoint file is "model.ckpt-2000.*" and the date is 20200611, then the suffix is "2000_20200611000000."


### Tips for QAT

The following are some tips for QAT.

- **Keras Model:** For Keras models, set `backend.set_learning_phase(1)` before creating the float train graph, and set `backend.set_learning_phase(0)` before creating the float evaluation graph. Moreover, `backend.set_learning_phase()` should be called after `backend.clear_session()`. TensorFlow 1.x QAT APIs are designed for TensorFlow native training APIs. Using Keras `model.fit()` APIs in QAT might lead to some "nodes not executed" issues. It is recommended to use QAT APIs in the TensorFlow 2 quantization tool with Keras APIs.

- **Dropout**: Experiments show that QAT works better without dropout ops. This tool does not support finetuning with dropouts at the moment and they should be removed or disabled before running QAT. This can be done by setting `is_training=false` when using tf.layers or call `tf.keras.backend.set_learning_phase(0)` when using tf.keras.layers.

- **Hyper-parameter**: QAT is like float model training/finetuning, so the techniques for float model training/finetuning are also needed. The optimizer type and the learning rate curve are some important parameters to tune.

## 4.2.2.4: vai_q_tensorflow Supported Operations and APIs

The following table lists the supported operations and APIs for vai_q_tensorflow.

**Supported Operations and APIs for vai_q_tensorflow**

|Type|Operation Type|tf.nn|tf.layers|tf.keras.layers|
| :--- | :--- | :--- | :--- | :--- |
|Convolution|Conv2D <br> DepthwiseConv2dNative|atrous_conv2d <br> conv2d <br> conv2d_transpose <br> depthwise_conv2d_native <br> separable_conv2d|Conv2D <br> Conv2DTranspose <br> SeparableConv2D|Conv2D <br> Conv2DTranspose <br> DepthwiseConv2D <br> SeparaleConv2D|
|Fully Connected|MatMul|/|Dense|Dense|
|BiasAdd|BiasAdd <br> Add|bias_add|/|/|
|Pooling|AvgPool <br> Mean <br> MaxPool|avg_pool <br> max_pool|AveragePooling2D <br> MaxPooling2D|AveragePooling2D <br> MaxPooling2D|
|Activation|ReLU <br> ReLU6 <br> Sigmoid <br> Swish <br> Hard-sigmoid <br> Hard-swish|relu <br> relu6 <br> leaky_relu <br> swish|/|ReLU <br> Leaky ReLU|
|BatchNorm[#1]|FusedBatchNorm|batch_normalization <br> batch_norm_with_glob <br> al_normalization <br> fused_batch_norm|BatchNormalization|BatchNormalization|
|Upsampling|ResizeBilinear <br> ResizeNearestNeighbor|/|/|UpSampling2D|
|Concat|Concat <br> ConcatV2|/|/|Concatenate|
|Others|Placeholder <br> Const <br> Pad <br> Squeeze <br> Reshape <br> ExpandDims|dropout[#2] <br> softmax[#3]|Dropout[#2] <br> Flatten|Input <br> Flatten <br> Reshape <br> Zeropadding2D <br> Softmax|

**Notes**:  
#1. Only supports Conv2D/DepthwiseConv2D/Dense+BN. BN is folded to speed up inference.  
#2. Dropout is deleted to speed up inference.  
#3. vai_q_tensorflow does not quantize the softmax output.  

## 4.2.2.5: vai_q_tensorflow Usage

The options supported by vai_q_tensorflow are shown in the following tables.

**vai_q_tensorflow Options**
<escape>
<table>
<thead>
  <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <td colspan="3"><b>Common Configuration</b></td>
  </tr>
  <tr>
    <td>--input_frozen_graph</td>
    <td>String</td>
    <td>TensorFlow frozen inference GraphDef file for the floating-point model. It is used for quantize calibration.</td>
  </tr>
  <tr>
    <td>--input_nodes</td>
    <td>String</td>
    <td>Specifies the name list of input nodes of the quantize graph, used together with <i>–output_nodes</i>, comma separated. Input nodes and output nodes are the start and end points of quantization. The subgraph between them is quantized if it is quantizable.<br><br><b>RECOMMENDED</b>: Set <i>–input_nodes</i> as the last nodes for preprocessing and `–output_nodes` as the last nodes for post-processing because some of the operations required for pre- and post-processing are not quantizable and might cause errors when compiled by the Vitis AI compiler if you need to deploy the quantized model to the DPU. The input nodes might not be the same as the placeholder nodes of the graph.</td>
  </tr>
  <tr>
    <td>--output_nodes</td>
    <td>String</td>
    <td>Specifies the name list of output nodes of the quantize graph, used together with <i>–input_nodes</i>, comma separated. Input nodes and output nodes are the start and end points of quantization. The subgraph between them is quantized if it is quantizable.<br><br><b>RECOMMENDED</b>: Set <i>–input_nodes</i> as the last nodes for preprocessing and <i>–output_nodes</i> as the last nodes for post-processing because some of the operations required for pre- and post-processing are not quantizable and might cause errors when compiled by the Vitis AI compiler if you need to deploy the quantized model to the DPU.</td>
  </tr>
  <tr>
    <td>--input_shapes</td>
    <td>String</td>
    <td>Specifies the shape list of input nodes. Must be a 4-dimension shape for each node, comma separated, for example 1,224,224,3; support unknown size for batch_size, for example ?,224,224,3. In case of multiple input nodes, assign the shape list of each node separated by :, for example, ?,224,224,3:?,300,300,1.</td>
  </tr>
  <tr>
    <td>--input_fn</td>
    <td>String</td>
    <td>Provides the input data for the graph used with the calibration dataset. The function format is `module_name.input_fn_name</i> (for example, <i>my_input_fn.input_fn</i>). The <i>-input_fn</i> should take an int object as input which indicates the calibration step, and should return a dict`(placeholder_node_name, numpy.Array)` object for each call, which is then fed into the placeholder operations of the model. For example, assign <i>–input_fn</i> to <i>my_input_fn.calib_input</i>, and write calib_input function in <i>my_input_fn.py</i> as:<br><br><i>def calib_input_fn:<br>&nbsp;&nbsp;&nbsp;&nbsp;# read image and do some preprocessing<br>&nbsp;&nbsp;&nbsp;&nbsp;return {“placeholder_1”: input_1_nparray,“placeholder_2”: input_2_nparray}</i><br><br><b>Note</b>: You do not need to do in-graph pre-processing again in input_fn because the subgraph before <i>-input_nodes</i> remains during quantization. Remove the pre-defined input functions (including default and random) because they are not commonly used. The preprocessing part which is not in the graph file should be handled in the input_fn.</td>
  </tr>
  <tr  align="center">
    <td colspan="3"><b>Quantize Configuration</b></td>
  </tr>
  <tr>
    <td>--weight_bit</td>
    <td>Int32</td>
    <td>Specifies the bit width for quantized weight and bias.<br>Default: 8</td>
  </tr>
  <tr>
    <td>--activation_bit</td>
    <td>Int32</td>
    <td>Specifies the bit width for quantized activation.<br>Default: 8</td>
  </tr>
  <tr>
    <td>--nodes_bit</td>
    <td>String</td>
    <td>Specifies the bit width of nodes. Node names and bit widths form a pair of parameters joined by a colon; the parameters are comma separated. When specifying the conv op name, only <i>vai_q_tensorflow</i> will quantize the weights of conv op using the specified bit width. For example, 'conv1/Relu:16,conv1/weights:8,conv1:16'.</td>
  </tr>
  <tr>
    <td>--method</td>
    <td>Int32</td>
    <td>Specifies the method for quantization.<br><br>0: Non-overflow method in which no values are saturated during quantization. Sensitive to outliers.<br>1: Min-diffs method that allows saturation for quantization to get a lower quantization difference. Higher tolerance to outliers. Usually ends with narrower ranges than the non-overflow method.<br>2: Min-diffs method with strategy for depthwise. It allows saturation for large values during quantization to get smaller quantization errors. A special strategy is applied for depthwise weights. It is slower than method 0 but has higher endurance to outliers.<br><br>Default value: 1</td>
  </tr>
  <tr>
    <td>--nodes_method</td>
    <td>String</td>
    <td>Specifies the method of nodes. Node names and method form a pair of parameters joined by a colon; the parameter pairs are comma separated. When specifying the conv op name, only <i>vai_q_tensorflow</i> will quantize weights of conv op using the specified method, for example, 'conv1/Relu:1,depthwise_conv1/weights:2,conv1:1'.</td>
  </tr>
  <tr>
    <td>--calib_iter</td>
    <td>Int32</td>
    <td>Specifies the iterations of calibration. Total number of images for calibration = calib_iter * batch_size.<br>Default value: 100</td>
  </tr>
  <tr>
    <td>--ignore_nodes</td>
    <td>String</td>
    <td>Specifies the name list of nodes to be ignored during quantization. Ignored nodes are left unquantized during quantization.</td>
  </tr>
  <tr>
    <td>--skip_check</td>
    <td>Int32</td>
    <td>If set to 1, the check for float model is skipped. Useful when only part of the input model is quantized.<br>Range: [0, 1]<br>Default value: 0</td>
  </tr>
  <tr>
    <td>--align_concat</td>
    <td>Int32</td>
    <td>Specifies the strategy for the alignment of the input quantize position for concat nodes.<br><br>0: Aligns all the concat nodes<br>1: Aligns the output concat nodes<br>2: Disables alignment<br><br>Default value: 0</td>
  </tr>
  <tr>
    <td>--simulate_dpu</td>
    <td>Int32</td>
    <td>Set to 1 to enable the simulation of the DPU. The behavior of DPU for some operations is different from TensorFlow. For example, the dividing in LeakyRelu and AvgPooling are replaced by bit-shifting, so there might be a slight difference between DPU outputs and CPU/GPU outputs. The vai_q_tensorflow quantizer simulates the behavior of these operations if this flag is set to 1.<br>Range: [0, 1]<br>Default value: 1</td>
  </tr>
  <tr>
    <td>--adjust_shift_bias</td>
    <td>Int32</td>
    <td>Specifies the strategy for shift bias check and adjustment for DPU compiler.<br><br>0: Disables shift bias check and adjustment<br>1: Enables with static constraints<br>2: Enables with dynamic constraints<br><br>Default value: 1</td>
  </tr>
  <tr>
    <td>--adjust_shift_cut</td>
    <td>Int32</td>
    <td>Specifies the strategy for shift cut check and adjustment for DPU compiler.<br><br>0: Disables shift cut check and adjustment<br>1: Enables with static constraints<br><br>Default value: 1</td>
  </tr>
  <tr>
    <td>--arch_type</td>
    <td>String</td>
    <td>Specifies the arch type for fix neuron. 'DEFAULT' means quantization range of both weights and activations are [-128, 127]. 'DPUCADF8H' means weights quantization range is [-128, 127] while activation is [-127, 127]</td>
  </tr>
  <tr>
    <td>--output_dir</td>
    <td>String</td>
    <td>Specifies the directory in which to save the quantization results.<br>Default value: “./quantize_results”</td>
  </tr>
  <tr>
    <td>--max_dump_batches</td>
    <td>Int32</td>
    <td>Specifies the maximum number of batches for dumping.<br>Default value: 1</td>
  </tr>
  <tr>
    <td>--dump_float</td>
    <td>Int32</td>
    <td>If set to 1, the float weights and activations are dumped.<br>Range: [0, 1]<br>Default value: 0</td>
  </tr>
  <tr>
    <td>--dump_input_tensors</td>
    <td>String</td>
    <td>Specifies the input tensor name of Graph when graph entrance is not a placeholder. Add a placeholder to the <i>dump_input_tensor</i>, so that `input_fn` can feed data.</td>
  </tr>
  <tr>
    <td>--scale_all_avgpool</td>
    <td>Int32</td>
    <td>Set to 1 to enable scale output of AvgPooling op to simulate DPU. Only kernel_size &lt;= 64 will be scaled. This operation does not affect the special case such as kernel_size=3,5,6,7,14<br>Default value: 1</td>
  </tr>
  <tr>
    <td>--do_cle</td>
    <td>Int32</td>
    <td><br>1: Enables implement cross layer equalization to adjust the weights distribution<br>0: Skips cross layer equalization operation<br><br>Default value: 0</td>
  </tr>
  <tr>
    <td>--replace_relu6</td>
    <td>Int32</td>
    <td>Available only for do_cle=1<br><br>1: Allows you to ReLU6 with ReLU<br>0: Skips replacement.<br><br>Default value: 1</td>
  </tr>
  <tr align="center">
    <td colspan="3"><b>Session Configurations</b></td>
  </tr>
  <tr>
    <td>--gpu</td>
    <td>String</td>
    <td>Specifies the IDs of the GPU device used for quantization separated by commas.</td>
  </tr>
  <tr>
    <td>--gpu_memory_fraction</td>
    <td>float</td>
    <td>Specifies the GPU memory fraction used for quantization, between 0-1.<br>Default value: 0.5</td>
  </tr>
  <tr align="center">
    <td colspan="3"><b>Others</b></td>
  </tr>
  <tr>
    <td>--help</td>
    <td></td>
    <td>Shows all available options of vai_q_tensorflow.</td>
  </tr>
  <tr>
    <td>--version</td>
    <td></td>
    <td>Shows the version information for vai_q_tensorflow.</td>
  </tr>
</tbody>
</table>
</escape>

### Examples

```shell
#show help: 
$vai_q_tensorflow --help

#quantize:
$vai_q_tensorflow quantize --input_frozen_graph frozen_graph.pb \
--input_nodes inputs \
--output_nodes predictions \
--input_shapes ?,224,224,3 \
--input_fn my_input_fn.calib_input

#dump quantized model:
$vai_q_tensorflow dump --input_frozen_graph quantize_results/quantize_eval_model.pb \
--input_fn my_input_fn.dump_input
```
Refer to [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo) for more TensorFlow model quantization examples.

## 4.2.3 Quantize ONNX Models

The Vitis AI Quantizer for ONNX models is customized based on [Quantization Tool](https://github.com/microsoft/onnxruntime/tree/rel-1.14.0/onnxruntime/python/tools/quantization) in ONNX Runtime.


### 4.2.3.1 Test Environment

* Python 3.7, 3.8
* ONNX>=1.12.0
* ONNX Rumtime>=1.14.0
* onnxruntime-extensions>=0.4.2

### 4.2.3.2 Installation 

You can install vai_q_onnx as follows:

#### Install from Source Code with Wheel Package
To build vai_q_onnx, run the following command:
```
$ sh build.sh
$ pip install pkgs/*.whl
```


### 4.2.3.3 Post Training Quantization (PTQ) 

The static quantization method first runs the model using a set of inputs called calibration data. During these runs, the quantization parameters for each activation are computed. These quantization parameters are written as constants to the quantized model and used for all inputs. Our quantization tool supports the following calibration methods: MinMax, Entropy and Percentile, and MinMSE.

```python
import vai_q_onnx

vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.VitisQuantFormat.FixNeuron,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE)
```

**Arguments**

* **model_input**: (String) Represents the file path of the model to be quantized.
* **model_output**: (String) Represents the file path where the quantized model is saved.
* **calibration_data_reader**: (Object or None) Calibration data reader. It enumerates the calibration data and generates inputs for the original model. If you want to use random data for a quick test, you can set calibration_data_reader to None. The default value is None.
* **quant_format**: (String) Specifies the quantization format of the model. It has the following options:
<br>**QOperator:** This option quantizes the model directly using quantized operators.
<br>**QDQ:** This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization only.
<br>**VitisQuantFormat.QDQ:** This option quantizes the model by inserting VAIQuantizeLinear/VAIDeQuantizeLinear into the tensor. It supports a wider range of bit-widths and configurations.
<br>**VitisQuantFormat.FixNeuron:** This option quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor.
* **calibrate_method**: (String) For DPU devices, set calibrate_method to either 'vai_q_onnx.PowerOfTwoMethod.NonOverflow' or 'vai_q_onnx.PowerOfTwoMethod.MinMSE' to apply power-of-2 scale quantization. The PowerOfTwoMethod currently supports two methods: MinMSE and NonOverflow. The default method is MinMSE.


### 4.2.3.4 Running vai_q_onnx


Quantization in ONNX Runtime refers to the linear quantization of an ONNX model. We have developed the vai_q_onnx tool as a plugin for ONNX Runtime to support more post-training quantization(PTQ) functions for quantizing a deep learning model. Post-training quantization (PTQ) is a technique to convert a pre-trained float model into a quantized model with little degradation in model accuracy. A representative dataset is needed to run a few batches of inference on the float model to obtain the distributions of the activations, which is also called quantized calibration.


Usage of vai_q_onnx supports is as follows:

#### vai_q_onnx Post-Training Quantization (PTQ)

Use the following steps to run PTQ with vai_q_onnx:

1.  ##### Preparing the Float Model and Calibration Set

Before running vai_q_onnx, ensure to prepare the float model and calibration set, including the files listed in the following table.

Table 1. Input files for vai_q_onnx

| No. | Name | Description |
| ------ | ------ | ----- |
| 1 | float model | Floating-point ONNX models in onnx format. |
| 2 | calibration dataset | A subset of the training dataset or validation dataset to represent the input data distribution, usually 100 to 1000 images are enough. |

2. ##### (Recommended) Pre-processing the Float Model

Pre-processing float32 model transforms and prepares it for quantization. It consists of the
following three optional steps:
* Symbolic shape inference: It is best suited for transformer models.
* Model Optimization: It uses ONNX Runtime native library to rewrite the computation graph, including merging computation nodes, and eliminating redundancies to improve runtime efficiency.
* ONNX shape inference.

The primary objective of these steps is to enhance quantization quality. The ONNX Runtime quantization tool performs optimally when the tensor's shape is known. Both symbolic shape inference and ONNX shape inference play a crucial role in determining tensor shapes. Symbolic shape inference is particularly effective for transformer-based models, whereas ONNX shape inference works well with other models. 
Model optimization performs certain operator fusion, making the quantization tool’s job easier. For instance, a Convolution operator followed by BatchNormalization can be fused into one during the optimization, which enables effective quantization. 
ONNX Runtime has a known issue: model optimization cannot output a model size greater than 2 GB. As a result, for large models, optimization must be skipped.
Pre-processing API is in the Python module onnxruntime.quantization.shape_inference, function quant_pre_process().
Pre-processing API can be found in the onnxruntime.quantization.shape_inference Python
module inside the quant_pre_process() function:

```python
from onnxruntime.quantization import shape_inference

shape_inference.quant_pre_process(
     input_model_path: str,
    output_model_path: str,
    skip_optimization: bool = False,
    skip_onnx_shape: bool = False,
    skip_symbolic_shape: bool = False,
    auto_merge: bool = False,
    int_max: int = 2**31 - 1,
    guess_output_rank: bool = False,
    verbose: int = 0,
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = False,
    external_data_location: str = "./",
    external_data_size_threshold: int = 1024,)
```

**Arguments**

* **input_model_path**: (String) Specifies the file path of the input model to be pre-processed for quantization.
* **output_model_path**: (String) Specifies the file path where the pre-processed model is saved.
* **skip_optimization**:  (Boolean) Indicates whether to skip the model optimization step. If set to True, model optimization is skipped, which may cause ONNX shape inference failure for some models. The default value is False.
* **skip_onnx_shape**:  (Boolean) Indicates whether to skip the ONNX shape inference step. The symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization because a tensor with an unknown shape cannot be quantized. The default value is False.
* **skip_symbolic_shape**:  (Boolean) Indicates whether to skip the symbolic shape inference step. Symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization because a tensor with an unknown shape cannot be quantized. The default value is False.
* **auto_merge**: (Boolean) Determines whether to automatically merge symbolic dimensions when a conflict occurs during symbolic shape inference. The default value is False.
* **int_max**:  (Integer) Specifies the maximum integer value that is to be considered as boundless for operations like slice during symbolic shape inference. The default value is 2**31 - 1.
* **guess_output_rank**: (Boolean) Indicates whether to guess the output rank to be the same as input 0 for unknown operations. The default value is False.
* **verbose**: (Integer) Controls the level of detailed information logged during inference. A value of 0 turns off logging, 1 logs warnings, and 3 logs detailed information. The default value is 0.
* **save_as_external_data**: (Boolean) Determines whether to save the ONNX model to external data. The default value is False.
* **all_tensors_to_one_file**: (Boolean) Indicates whether to save all the external data to one file. The default value is False.
* **external_data_location**: (String) Specifies the file location where the external file is saved. The default value is "./".
* **external_data_size_threshold**:  (Integer) Specifies the size threshold for external data. The default value is 1024.

3.  ##### Quantizing Using the vai_q_onnx API
The static quantization method first runs the model using a set of inputs called calibration data. During these runs, the quantization parameters are computed for each activation. These quantization parameters are written as constants to the quantized model and used for all inputs. Vai_q_onnx quantization tool has expanded calibration methods to power-of-2 scale/float scale quantization methods. Float scale quantization methods include MinMax, Entropy, and Percentile. Power-of-2 scale quantization methods include MinMax and MinMSE:

```python

vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.VitisQuantFormat.FixNeuron,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    input_nodes=[],
    output_nodes=[],
    extra_options=None,)
```


**Arguments**

* **model_input**: (String) Specifies the path of the model to be quantized.
* **model_output**: (String) Specifies the file path where the quantized model is saved.
* **calibration_data_reader**: (Object or None) Calibration data reader that enumerates the calibration data and generates inputs for the original model. If you want to use random data for a quick test, you can set calibration_data_reader to None.
* **quant_format**: (Enum) Defines the quantization format for the model. It has the following options:
<br>**QOperator**: This option quantizes the model directly using quantized operators.
<br>**QDQ**: This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization.
<br>**VitisQuantFormat.QDQ** This option quantizes the model by inserting VAIQuantizeLinear/VAIDeQuantizeLinear into the tensor. It supports a wider range of bit-widths and configurations.
<br>**VitisQuantFormat.FixNeuron** This option quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor. This is the default value.
* **calibrate_method**:  (Enum) Used to set the power-of-2 scale quantization method for DPU devices. It currently supports two methods: 'vai_q_onnx.PowerOfTwoMethod.NonOverflow' and 'vai_q_onnx.PowerOfTwoMethod.MinMSE'. The default value is 'vai_q_onnx.PowerOfTwoMethod.MinMSE'.
* **input_nodes**:  (List of Strings) List of the names of the starting nodes to be quantized. The nodes before these start nodes in the model are not optimized or quantized. For example, this argument can be used to skip some pre-processing nodes or stop quantizing the first node. The default value is [].
* **output_nodes**: (List of Strings) Names of the end nodes to be quantized. The nodes after these nodes in the model are not optimized or quantized. For example, this argument can be used to skip some post-processing nodes or stop quantizing the last node. The default value is [].
* **extra_options**: (Dict or None) Dictionary of additional options that can be passed to the quantization process. If there are no additional options to provide, this can be set to None. The default value is None.


4.  ##### (Optional) Evaluating the Quantized Model
If you have scripts to evaluate float models, like the models in AMD Model Zoo, you can replace the float model file with the quantized model for evaluation.

To support the customized FixNeuron op, the vai_dquantize module should be imported. THe following is example:

```python
import onnxruntime as ort
from vai_q_onnx.operators.vai_ops.qdq_ops import vai_dquantize

so = ort.SessionOptions()
so.register_custom_ops_library(_lib_path())
sess = ort.InferenceSession(dump_model, so)
input_name = sess.get_inputs()[0].name
results_outputs = sess.run(None, {input_name: input_data})
```

After that, evaluate the quantized model just as the float model.


5.  ##### (Optional) Dumping the Simulation Results
Sometimes, after deploying the quantized model, it is essential to compare the simulation results on the CPU and GPU with the output values on the DPU.
You can use the dump_model API of vai_q_onnx to dump the simulation results with the quantized_model:

```python
# This function dumps the simulation results of the quantized model,
# including weights and activation results.
vai_q_onnx.dump_model(
    model,
    dump_data_reader=None,
    output_dir='./dump_results',
    dump_float=False)
```

**Arguments**

* **model**: (String) Specifies the file path of the quantized model whose simulation results are to be dumped.
* **dump_data_reader**:  (Object or None) Data reader that is used for the dumping process. It generates inputs for the original model. 
* **output_dir**: (String) Specifies the directory where the dumped simulation results are saved. After successful execution of the function, dump results are generated in this specified directory. The default value is './dump_results'.
* **dump_float**: (Boolean) Determines whether to dump the floating-point value of weights and activation results. If set to True, the float values are dumped. The default value is False.

**Note**: The batch_size of the dump_data_reader should be set to 1 for DPU debugging.

After successfully executing the command, the dump results are generated in the output_dir.
Each quantized node's weights and activation results are saved separately in *.bin and *.txt formats. In cases where the node output is not quantized, such as the softmax node, the float activation results are saved in *_float.bin and *_float.txt formats if the option "save_float" is set to True.
The following table shows an example of the dump results.

Table 2. Example of Dumping Results

| Quantized | Node Name | Saved Weights and Activations|
| ------ | ------ | ----- |
| Yes | resnet_v1_50_conv1 | {output_dir}/dump_results/quant_resnet_v1_50_conv1.bin<br>{output_dir}/dump_results/quant_resnet_v1_50_conv1.txt|
| Yes | resnet_v1_50_conv1_weights | {output_dir}/dump_results/quant_resnet_v1_50_conv1_weights.bin<br>{output_dir}/dump_results/quant_resnet_v1_50_conv1_weights.txt  |
| No | resnet_v1_50_softmax | {output_dir}/dump_results/quant_resnet_v1_50_softmax_float.bin<br>{output_dir}/dump_results/quant_resnet_v1_50_softmax_float.txt |


### 4.2.3.5 List of Vai_q_onnx Supported Quantized Ops

The following table lists the supported operations and APIs for vai_q_onnx.

Table 3. List of Vai_q_onnx Supported Quantized Ops
| supported ops | Comments |
| :-- | :-- |
| Add| |
| Conv| |
| ConvTranspose| |
| Gemm| |
| Concat| |
| Relu| |
| Reshape| |
| Transpose| |
| Resize| |
| MaxPool| |
| GlobalAveragePool| |
| AveragePool| |
| MatMul| |
| Mul| |
| Sigmoid| |
| Softmax| |


### 4.2.3.6 vai_q_onnx APIs

quantize_static Method

```python
vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.VitisQuantFormat.FixNeuron,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    input_nodes=[],
    output_nodes=[],
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    optimize_model=True,
    use_external_data_format=False,
    extra_options=None,)
```


**Arguments**


* **model_input**: (String) Specifies the file path of the model that is to be quantized.
* **model_output**: (String) Specifies the file path where the quantized model will be saved.
* **calibration_data_reader**: (Object or None) Calibration data reader that enumerates the calibration data and generates inputs for the original model. If you want to use random data for a quick test, you can set calibration_data_reader to None.
* **quant_format**: (Enum) Defines the quantization format for the model. It has the following options:
<br>**QOperator** Quantizes the model directly using quantized operators.
<br>**QDQ** Quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization only.
<br>**VitisQuantFormat.QDQ** Quantizes the model by inserting VAIQuantizeLinear/VAIDeQuantizeLinear into the tensor. It supports a wider range of bit-widths and configurations.
<br>**VitisQuantFormat.FixNeuron** Quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor. This is the default value.
* **calibrate_method**:  (Enum) Used to set the power-of-2 scale quantization method for DPU devices. It currently supports two methods: 'vai_q_onnx.PowerOfTwoMethod.NonOverflow' and 'vai_q_onnx.PowerOfTwoMethod.MinMSE'. The default value is 'vai_q_onnx.PowerOfTwoMethod.MinMSE'.
* **input_nodes**:  (List of Strings) Names of the starting nodes to be quantized. Nodes in the model before these nodes will not be quantized. For example, this argument can be used to skip some pre-processing nodes or stop the first node from being quantized. The default value is an empty list ([]).
* **output_nodes**: (List of Strings) Names of the end nodes to be quantized. Nodes in the model after these nodes are not be quantized. For example, this argument can be used to skip some post-processing nodes or stop the last node from being quantized. The default value is an empty list ([]).
* **op_types_to_quantize**:  (List of Strings or None) If specified, only operators of the given types are quantized (For example, ['Conv'] to only quantize Convolutional layers). By default, all supported operators are quantized.
* **per_channel**: (Boolean) Determines whether weights should be quantized per channel. For DPU devices, this must be set to False as they currently do not support per-channel quantization.
* **reduce_range**: (Boolean) If True, quantizes weights with 7-bits. For DPU devices, this must be set to False as they currently do not support reduced range quantization.
* **activation_type**: (QuantType) Specifies the quantization data type for activations. For DPU devices, this must be set to QuantType.QInt8. For more details on data type selection, refer to the ONNX Runtime quantization documentation.
* **weight_type**: (QuantType) Specifies the quantization data type for weights. For DPU devices, this must be set to QuantType.QInt8.
* **nodes_to_quantize**:(List of Strings or None) If specified, only the nodes in this list are quantized. The list should contain the names of the nodes, for example, ['Conv__224', 'Conv__252'].
* **nodes_to_exclude**:(List of Strings or None) If specified, the nodes in this list are excluded from quantization.
* **optimize_model**:(Boolean) If True, optimizes the model before quantization. However, this is not recommended as optimization changes the computation graph, making the debugging of quantization loss difficult.
* **use_external_data_format**:  (Boolean) Used for large size (>2GB) model. The default is False.
* **extra_options**:  (Dictionary or None) Contains key-value pairs for various options in different cases. Current used:<br>
                **extra.Sigmoid.nnapi = True/False**  (Default is False)
                <br>**ActivationSymmetric = True/False**: If True, calibration data for activations is symmetrized. The default is False. When using PowerOfTwoMethod for calibration, this should always be set to True.
                <br>**WeightSymmetric = True/False**: If True, calibration data for weights is symmetrized. The default is True. When using PowerOfTwoMethod for calibration, this should always be set to True.
                <br>**EnableSubgraph = True/False**:  If True, the subgraph is quantized. The default is False. 
                <br>**ForceQuantizeNoInputCheck = True/False**:
                    If True, latent operators such as maxpool and transpose are always quantize their inputs, generating quantized outputs even if their inputs have not been quantized. The default behavior can be overridden for specific nodes using nodes_to_exclude.
                <br>**MatMulConstBOnly = True/False**:
                    If True, only MatMul operations with a constant 'B' is quantized. The default is False for static mode.
                <br>**AddQDQPairToWeight = True/False**:
                     If True, both QuantizeLinear and DeQuantizeLinear nodes are inserted for weight, maintaining its floating-point format. The default is False, which quantizes floating-point weight and feeds it solely to an inserted DeQuantizeLinear node. In the PowerOfTwoMethod calibration method, QDQ should always appear as a pair, hence this should be set to True.
                <br>**OpTypesToExcludeOutputQuantization = list of op type**:
                     If specified, the output of operators with these types is not quantized. The default is an empty list.
                <br>**DedicatedQDQPair = True/False**: If True, an identical and dedicated QDQ pair is created for each node. The default is False, allowing multiple nodes to share a single QDQ pair as their inputs.
                <br>**QDQOpTypePerChannelSupportToAxis = dictionary**:
                     Sets the channel axis for specific operator types (e.g., {'MatMul': 1}). This is only effective when per-channel quantization is supported and per_channel is True. If a specific operator type supports per-channel quantization but no channel axis is explicitly specified, the default channel axis is used. For DPU devices, this must be set to {} as per-channel quantization is currently unsupported.
                <br>**CalibTensorRangeSymmetric = True/False**:
                    If True, the final range of the tensor during calibration is symmetrically set around the central point "0". The default is False. In PowerOfTwoMethod calibration method, this should always be set to True.
                <br>**CalibMovingAverage = True/False**:
                    If True, the moving average of the minimum and maximum values is computed when the calibration method selected is MinMax. The default is False. In PowerOfTwoMethod calibration method, this should be set to False.
                <br>**CalibMovingAverageConstant = float**:
                    Specifies the constant smoothing factor to use when computing the moving average of the minimum and maximum values. The default is 0.01. This is only effective when the calibration method selected is MinMax and CalibMovingAverage is set to True. In PowerOfTwoMethod calibration method, this option is unsupported.


[< Previous](/docs/4_deploy_your_own_model/prune_model/prunemodel.md) | [Next >](/docs/4_deploy_your_own_model/deploy_model/deployingmodel.md)

<hr/>

# License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.
