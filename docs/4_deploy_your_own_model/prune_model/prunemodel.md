<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.2 User Guide </h1>
    </td>
 </tr>
 <tr>
 <td align="center"><h1>Step 4.1: Prune Model with UIF Optimizer (Optional)</h1>
 </td>
 </tr>
</table>

# Table of Contents
- [4.1.1: Pruning](#411-pruning)
- [4.1.2: UIF Optimizer Overview](#412-uif-optimizer-overview)
- [4.1.3: Prune Model with UIF Optimizer PyTorch](#413-prune-model-with-uif-optimizer-pytorch)
  - [4.1.3.1: Coarse-grained Pruning](#4131-coarse-grained-pruning)
  - [4.1.3.2: Once-for-All (OFA)](#4132-once-for-all-ofa)
- [4.1.4: Prune Model with UIF Optimizer TensorFlow](#414-prune-model-with-uif-optimizer-tensorflow)
  - [4.1.4.1: Creating a Baseline Model](#4141-creating-a-baseline-model)
  - [4.1.4.2: Creating a Pruning Runner](#4142-creating-a-pruning-runner)
  - [4.1.4.3: Pruning the Baseline Model](#4143-pruning-the-baseline-model)
  - [4.1.4.4: Fine Tuning a Spare Model](#4144-fine-tuning-a-sparse-model)
  - [4.1.4.5: Performing Iterative Pruning](#4145-performing-iterative-pruning)
  - [4.1.4.6: Getting the Pruned Model](#4146-getting-the-pruned-model)

  _Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._


# 4.1.1: Pruning

Neural networks are typically over-parameterized with significant redundancy. Pruning is the process of eliminating redundant weights while keeping the accuracy loss to a minimum. Industry research has led to several techniques that serve to reduce the computational cost of neural networks for inference. These techniques include:

- Fine-grained pruning
- Coarse-grained pruning
- Neural Architecture Search (NAS)

The simplest form of pruning is called fine-grained pruning and results in sparse matrices (that is, matrices that have many elements equal to zero), which require the addition of specialized hardware and techniques for weight skipping and compression. Fine-grained pruning is not currently supported by UIF Optimizer. 

UIF Optimizer employs coarse-grained pruning, which eliminates neurons that do not contribute significantly to the accuracy of the network. For convolutional layers, the coarse-grained method prunes the entire 3D kernel and hence is also known as channel pruning. 
Inference acceleration can be achieved without specialized hardware for coarse-grained pruned models. Pruning always reduces the accuracy of the original model. Retraining (fine-tuning) adjusts the remaining weights to recover accuracy.

Coarse-grained pruning works well on large models with common convolutions, such as ResNet and VGGNet. However, for depth-wise convolution-based models such as MobileNet-v2, the accuracy of the pruned model drops dramatically even at small pruning rates. In addition to pruning, UIF Optimizer provides a one-shot neural architecture search (NAS) based approach to reduce the computational cost of inference (currently only available in Optimizer PyTorch).

# 4.1.2: UIF Optimizer Overview

Inference in machine learning is computationally intensive and requires high memory bandwidth to meet the low-latency and high-throughput requirements of various applications. UIF Optimizer provides the ability to prune neural network models. It prunes redundant kernels in neural networks, thereby reducing the overall computational cost for inference. The pruned models produced by UIF Optimizer are then quantized by UIF Quantizer to be further optimized.

The following tables show the features that are supported by UIF Optimizer for different frameworks:

<table>
  <colgroup span="3"></colgroup>
  <colgroup span="3"></colgroup>
  <tr>
	<th rowspan="2">Framework</th>
	<th rowspan="2">Versions</th>
    <th colspan="3" scope="colgroup">Features</th>
  </tr>
  <tr>
    <th scope="col">Iterative</th>
    <th scope="col">One-step</th>
	<th scope="col">OFA</th>
  </tr>
  <tr>
    <th scope="row">PyTorch</th>
    <td>Supports 1.7 - 1.13</td>
    <td>Yes</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <th scope="row">TensorFlow</th>
    <td>Supports 2.4 - 2.12</td>
    <td>Yes</td>
    <td>No</td>
    <td>No</td>
  </tr>
</table>

# 4.1.3: Prune Model with UIF Optimizer PyTorch

UIF Optimizer PyTorch provides three methods of model pruning:

- Iterative pruning
- One-step pruning
- Once-for-all (OFA)

Iterative pruning and one-step pruning belong to the coarse-grained pruning category, while the once-for-all technique is an NAS-based approach.

## 4.1.3.1 Coarse-grained Pruning

### Create the Baseline Model

For simplicity, ResNet18 from torchvision is used here. In real life applications, the process of creating a model can be complicated.

```python
from torchvision.models.resnet import resnet18
model = resnet18(pretrained=True)
```

### Create a Pruning Runner

Import modules and prepare input signature:

```python
from pytorch_nndct import get_pruning_runner

# The input signature should have the same shape and dtype as the model input.
input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32)
```
Create an iterative pruning runner:

```python
runner = get_pruning_runner(model, input_signature, 'iterative')
```

Alternatively, create a one-step pruning runner:

```python
runner = get_pruning_runner(model, input_signature, 'one_step')
```

### Pruning the Baseline Model

#### **Iterative Pruning**

The method includes two stages: model analysis and pruned model generation.

After the model analysis is completed, the analysis result is saved in the file named `.vai/your_model_name.sens`. You can prune a model iteratively using this file. In other words, prune the model to the target ratio gradually to avoid the failure to improve the model performance in the retraining stage that is caused by setting the pruning ratio too high.

1. Define an evaluation function. The function must take a model as its first argument and return a score.

```python
def eval_fn(model, dataloader):
  top1 = AverageMeter('Acc@1', ':6.2f')
  model.eval()
  with torch.no_grad():
  for i, (images, targets) in enumerate(dataloader):
    images = images.cuda()
    targets = targets.cuda()
    outputs = model(images)
    acc1, _ = accuracy(outputs, targets, topk=(1, 5))
    top1.update(acc1[0], images.size(0))
  return top1.avg
```
2. Run model analysis and get a pruned model.

```python
runner.ana(eval_fn, args=(val_loader,))
model = pruning_runner.prune(removal_ratio=0.2)
```
Run analysis only once for the same model. You can prune the model iteratively without re-running analysis because there is only one pruned model generated for a specific pruning ratio.

The subnetwork obtained by pruning may not be perfect because an approximate algorithm is used to generate this unique pruned model according to the analysis result.

The one-step pruning method can generate a better subnetwork.

#### One-step Pruning

The method also includes two stages: adaptive batch normalization (BN) based searching for pruning strategy and pruned model generation.
After searching, a file named `.vai/your_model_name.search` is generated in which the search result (pruning strategies and corresponding evaluation scores) is stored. You can get the final pruned model in one-step.

`num_subnet` provides the number of candidate subnetworks satisfying the sparsity requirement to be searched.
The best subnetwork can be selected from these candidates. The higher the value, the longer it takes to search, but the higher the probability of finding a better subnetwork.

```python
# Adaptive-BN-based searching for pruning strategy. 'calibration_fn' is a function for calibrating BN layer's statistics.
runner.search(gpus=['0'], calibration_fn=calibration_fn, calib_args=(val_loader,), eval_fn=eval_fn, eval_args=(val_loader,), num_subnet=1000, removal_ratio=0.7)
model = runner.prune(removal_ratio=0.7, index=None)
```

The `eval_fn` is the same with the iterative pruning method. A `calibration_fn` function that implements adaptive-BN is shown in the following example code. It is recommended to define your code in a similar way.

```python
def calibration_fn(model, dataloader, number_forward=100):
  model.train()
  with torch.no_grad():
    for index, (images, target) in enumerate(dataloader):
      images = images.cuda()
      model(images)
    if index > number_forward:
      break
```
The one-step pruning method has several advantages over the iterative approach:

- The generated pruned models are more accurate. All subnetworks that meet the requirements are evaluated.
- The workflow is simpler because you can obtain the final pruned model in one step without iterations.
- Retraining a slim model is faster than a sparse model.

There are two disadvantages to one-step pruning: one is that the random generation of pruning strategy is unstable, and the other is that the subnetwork searching must be performed once for every pruning ratio.

### Retraining the Pruned Model

Retraining the pruned model is the same as training a baseline model:

```python
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
best_acc1 = 0 

for epoch in range(args.epoches):
  train(train_loader, model, criterion, optimizer, epoch)
  acc1, acc5 = evaluate(val_loader, model, criterion)

  is_best = acc1 > best_acc1
  best_acc1 = max(acc1, best_acc1)
  if is_best:
    torch.save(model.state_dict(), 'model_pruned.pth')
    # Sparse model has one more special method in iterative pruning.
    if hasattr(model, 'slim_state_dict'):
      torch.save(model.slim_state_dict(), 'model_slim.pth')
```

## 4.1.3.2: Once-for-All (OFA)

Steps for the once-for-all method are as follows:

### Creating a Model

For simplicity, mobilenet_v2 from torchvision is used here.

```python
from torchvision.models.mobilenet import mobilenet_v2
model = mobilenet_v2(pretrained=True)
```

### Creating an OFA Pruner

The pruner requires two arguments:

- The model to be pruned
- The inputs needed by the model for inference

```python
import torch
from pytorch_nndct import OFAPruner

inputs = torch.randn([1, 3, 224, 224], dtype=torch.float32)
pruner = OFAPruner(model, inputs)
```
**Note:** The input does not need to be real data. You can use randomly generated dummy data if it has the same shape and type as the real data.

### Generating an OFA Model

Call `ofa_model()` to get an OFA model. This method finds all the `nn.Conv2d` / `nn.ConvTranspose2d`and `nn.BatchNorm2d` modules,
then replaces those modules with `DynamicConv2d` / `DynamicConvTranspose2d` and `DynamicBatchNorm2d`.

A list of pruning ratios is required to specify what the OFA model will be.

For each convolution layer in the OFA model, an arbitrary pruning ratio can be used in the output channel. The maximum and minimum values in this list represent the maximum and minimum compression rates of the model. Other values in the list represent the subnetworks to be optimized. By default, the pruning ratio is set to [0.5, 0.75, 1].

For a subnetwork sampled from the OFA model, the out channels of a convolution layer are one of the numbers in the pruning ratio list multiplied by its original number. For example, for a pruning ratio list of [0.5, 0.75, 1] and a convolution layer nn.Conv2d(16, 32, 5), the out channels of this layer in a sampled subnetwork are [0.5*32, 0.75*32, 1*32].

Because the first and last layers have a significant impact on network performance, they are commonly excluded from pruning.
By default, this method automatically identifies the first convolution and the last convolution, then puts them into the list of excludes. Setting `auto_add_excludes=False` can cancel this feature.

```python
ofa_model = ofa_pruner.ofa_model([0.5, 0.75, 1], excludes = None, auto_add_excludes=True)
```
### Training an OFA Model
This method uses the [sandwich rule](https://arxiv.org/abs/2003.11142) to jointly optimize all the OFA subnetworks. The `sample_random_subnet()` function can be used to get a subnetwork. The dynamic subnetwork can do a forward/backward pass.

In each training step, given a mini-batch of data, the sandwich rule samples a ‘max’ subnetwork, a ‘min’ subnetwork, and two random subnetworks. Each subnetwork does a separate forward/backward pass with the given data and then all the subnetworks update their parameters together.

```python
# using sandwich rule and sampling subnet.
for i, (images, target) in enumerate(train_loader):

  images = images.cuda(non_blocking=True)
  target = target.cuda(non_blocking=True)

  # total subnets to be sampled
  optimizer.zero_grad()

  teacher_model.train()
  with torch.no_grad():
    soft_logits = teacher_model(images).detach()

  for arch_id in range(4):
    if arch_id == 0:
      model, _ = ofa_pruner.sample_subnet(ofa_model,'max')
    elif arch_id == 1:
      model, _ = ofa_pruner.sample_subnet(ofa_model,'min')
    else:
      model, _ = ofa_pruner.sample_subnet(ofa_model,'random') 

    output = model(images)

    loss = kd_loss(output, soft_logits) + cross_entropy_loss(output, target) 
    loss.backward()

  torch.nn.utils.clip_grad_value_(ofa_model.parameters(), 1.0)
  optimizer.step()
  lr_scheduler.step()
```
### Searching Constrained Subnetworks

After the training is completed, you can conduct an [evolutionary search](https://arxiv.org/abs/1802.01548) based on the neural-network-twins to get a subnetwork with the best trade-offs between FLOPs and accuracy using a minimum and maximum FLOPs range.

```python
pareto_global = ofa_pruner.run_evolutionary_search(ofa_model, calibration_fn,
    (train_loader,) eval_fn, (val_loader,), 'acc1', 'max', min_flops=230, max_flops=250)
ofa_pruner.save_subnet_config(pareto_global, 'pareto_global.txt')
```
The searching result looks like the following:

```
{ 
"230": { 
    "net_id": "net_evo_0_crossover_0", 
    "mode": "evaluate",
    "acc1": 69.04999542236328,
    "flops": 228.356192,
    "params": 3.096728,
    "subnet_setting": [...]
}
"240": {
    "net_id": "net_evo_0_mutate_1",
    "mode": "evaluate",
    "acc1": 69.22000122070312,
    "flops": 243.804128,
    "params": 3.114,
    "subnet_setting": [...]
}}
```
### Getting a Subnetwork

Call `get_static_subnet()` to get a specific subnetwork. The `static_subnet` can be used for finetuning and quantization.

```python
pareto_global = ofa_pruner.load_subnet_config('pareto_global.txt')
static_subnet, static_subnet_config, flops, params = ofa_pruner.get_static_subnet(
    ofa_model, pareto_global['240']['subnet_setting'])
```
## 4.1.4: Prune Model with UIF Optimizer TensorFlow

**Note:** Only iterative pruning is supported with TensorFlow in this release of UIF.

UIF Optimizer TensorFlow only supports Keras models created by the [Functional API](https://www.tensorflow.org/guide/keras/functional/) or the [Sequential API](https://www.tensorflow.org/guide/keras/sequential_model).
[Subclassed](https://www.tensorflow.org/guide/keras/custom_layers_and_models) models are not supported.

### 4.1.4.1: Creating a Baseline Model

Here, a simple MNIST convnet from the [Keras vision example](https://keras.io/examples/vision/mnist_convnet) is used.

```python
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(), layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])
```
### 4.1.4.2: Creating a Pruning Runner

To create an input specification with shape and dtype and to use this specification to get a pruning runner, use the following command:

```python
from tf_nndct.optimization import IterativePruningRunner

input_shape = [28, 28, 1]
input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(model, input_spec)
```

### 4.1.4.3: Pruning the Baseline Model
To prune a model, follow these steps:

1. Define a function to evaluate model performance. The function must satisfy two requirements:

- The first argument must be a keras.Model instance to be evaluated.
- Returns a Python number to indicate the performance of the model.

```python
def evaluate(model):
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  score = model.evaluate(x_test, y_test, verbose=0)
  return score[1]
```
2. Use this evaluation function to run model analysis:

```python
runner.ana(evaluate)
```
3. Determine a pruning ratio. The ratio indicates the reduction in the amount of floating-point computation of the model in forward pass.

[MACs of pruned model] = (1 – ratio) * [MACs of original model]
The value of ratio should be in (0, 1):

```python
sparse_model = runner.prune(ratio=0.2)
```
**Note:** `ratio` is only an approximate target value and the actual pruning ratio may not be exactly equal to this value.

The returned model from `prune()` is sparse, which means that the pruned channels are set to zeros and model size remains unchanged.
The sparse model has been used in the iterative pruning process.
The sparse model is converted to a pruned dense model only after pruning is completed.

Besides returning a sparse model, the pruning runner generates a specification file in the `.vai` directory that describes how each layer is pruned.

### 4.1.4.4: Fine-tuning a Sparse Model

Training a sparse model is no different from training a normal model. The model maintains sparsity internally. There is no need for any additional actions other than adjusting the hyper-parameters.

```python
sparse_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
sparse_model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)
sparse_model.save_weights("model_sparse_0.2", save_format="tf")
```
**Note:** When calling `save_weights`, use the "tf" format to save the weights.

### 4.1.4.5: Performing Iterative Pruning

Load the checkpoint saved from the previous fine-tuning stage to the model. Increase the ratio value to get a sparser model.
Then continue to fine-tune this sparse model. Repeat this pruning and fine-tuning loop a couple of times until the sparsity reaches the desired value.

```python
model.load_weights("model_sparse_0.2")

input_shape = [28, 28, 1]
input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(model, input_spec)
sparse_model = runner.prune(ratio=0.5)
```
### 4.1.4.6: Getting the Pruned Model

When the iterative pruning is completed, a sparse model is generated, which has the same number of parameters as the original model but with many of them now set to zero.

Call `get_slim_model()` to remove zeroed parameters from the sparse model and retrieve the pruned model:

```python
model.load_weights("model_sparse_0.5")

input_shape = [28, 28, 1]
input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(model, input_spec)
runner.get_slim_model()
```

By default, the runner uses the latest pruning specification to generate the slim model. You can see what the latest specification file is with the following command:

```
$ cat .vai/latest_spec
$ ".vai/mnist_ratio_0.5.spec"
```
If this file does not match your sparse model, you can explicitly specify the file path to be used:

```python
runner.get_slim_model(".vai/mnist_ratio_0.5.spec")
```
<hr/>

[< Previous](/docs/3_run_example/runexample-migraphx.md) | [Next >](/docs/4_deploy_your_own_model/quantize_model/quantizemodel.md)

<hr/>

# License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.
