<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.2 User Guide </h1>
    </td>
 </tr>
 <tr>
 <td align="center"><h1>Step 4.4: Serve Model with Inference Server</h1>
 </td>
 </tr>
</table>

You can use the AMD Inference Server to serve your models.
The server enables you to make requests to your model directly or using REST and gRPC using a common API, independent of whether the model is using CPUs (with ZenDNN), GPUs (with MIGraphX), or FPGAs (with Vitis&trade; AI).
Use the [inference server installation instructions](/docs/1_installation/installation.md#14-get-the-inference-server-docker-image-for-model-serving) to get the Docker® images for the server.
For testing, you can use the development image and move to the deployment image when you are ready.
In all cases, you must configure your host machines for the appropriate hardware backends as described in the [UIF installation instructions](/docs/1_installation/installation.md), such as installing the ROCm™ platform for GPUs and XRT for FPGAs.
There are several methods you can use to serve your models with different benefits and tradeoffs, which are discussed here.


This UIF release uses AMD Inference Server 0.4.0. The full documentation for the server for this release is available [online](https://xilinx.github.io/inference-server/0.4.0/index.html).

The latest version of the server and documentation are available on [GitHub](https://github.com/Xilinx/inference-server).

# Table of Contents

- [4.4.1: Using the Development Image](#441-using-the-development-image)
- [4.4.2: Using the Deployment Image](#442-using-the-deployment-image)
- [4.4.3: Making Requests with HTTP or gRPC](#443-making-requests-with-http-or-grpc)

  _Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._

  
# 4.4.1: Using the Development Image

Using the development image to serve your model allows you to have greater control and visibility into its operation, which can be useful for debugging and analyzing. 
You must build the development image on a Linux host machine before using it.
The process to build and run the development image is documented in the Inference Server's [development quick start guide](https://xilinx.github.io/inference-server/main/quickstart_development.html).

The development container mounts your working directory inside so you can place the model you want to serve somewhere in that tree.
In the container, you can compile the server in debug or release configuration and start it.

Next, you need to load a worker to serve your model. The worker depends on the type of model you want to serve: CPU, GPU, or FPGA.
They use the `zendnn`, `migraphx`, and `xmodel` workers respectively.
At load-time, you can pass worker-specific arguments to configure how it behaves.
In particular, you pass the path to your model to the worker at load-time.
After the load succeeds, the server will respond with an endpoint string that you will need for subsequent requests.

## 4.4.1.1  Naming Format for *.mxr* Files

In UIF 1.2, the MIGraphX worker requires a naming format for *.mxr files that differs from the names used for Modelzoo in Section 2.  The required format is 
*\<model\>_bXX.mxr* where XX is the compiled model's batch size. For example, *resnet50_b32.mxr*.  If you use compiled *.mxr models that come from any source other than what the MIGraphX worker itself compiles, you must rename them to match this format. For example, 

```
    $ cp resnet50.mxr resnet50_b32.mxr
``` 

When requesting a worker, the _bXX suffix must be left out of the requested model name, so for this example the request would contain parameters *batch=32* and *model="resnet50.mxr"* or simply *model="resnet50"*.

## 4.4.1.2  Sending Server Requests
You can send requests to the server from inside or outside the container.

From inside the container, the default address for HTTP and gRPC is `http://127.0.0.1:8998` and `127.0.0.1:50051`. However, users can change the default address when starting the server.

- From the outside, the easiest approach is to use `docker ps` to list the ports the development container has exposed.
- From the same host machine, you can use `http://127.0.0.1:<port>` corresponding to the port listed that maps to 8998 in the container for HTTP requests.

  When you have the address and the endpoint returned by the load, you can [make requests](#443-making-requests-with-http-or-grpc) to the server.

# 4.4.2: Using the Deployment Image

The deployment image is a minimal image that contains a precompiled server executable that starts automatically when the container starts. This image is suitable for deployment with Docker, Kubernetes, or [KServe](https://xilinx.github.io/inference-server/0.4.0/kserve.html). With the latter two methods, you need to install and [set up a Kubernetes cluster](https://kubernetes.io/docs/setup/).

The process to build and run the deployment image is documented in the Inference Server's [deployment guide](https://xilinx.github.io/inference-server/0.4.0/deployment.html) and the [KServe deployment guide](https://xilinx.github.io/inference-server/0.4.0/kserve.html). As with the development image, you will need to load a worker to serve your model and use the endpoint it returns to make requests.


When the container is up, get the address for the server. With Docker, you can use `docker ps` to get the exposed ports.
With KServe, there is a [separate process](https://kserve.github.io/website/master/get_started/first_isvc/#3-check-inferenceservice-status) to determine the ingress address.

When you have the address and the endpoint returned by the load, you can [make requests](#443-making-requests-with-http-or-grpc) to the server.

# 4.4.3: Making Requests with HTTP or gRPC

Making requests to the server is most easily accomplished with the Python library. You can also use the C++ library equivalently, but this works best if you are using the development container because it has the library and all its dependencies already.

While you can use `curl` to query the status endpoints, making more complicated inference requests using `curl` can be difficult.
In the development container, the Python library is installed as part of the server compilation so you can use it from there.

To use the library elsewhere, you need to install it with pip:

```
    $ pip install amdinfer
``` 

More detailed information and discussion around making requests using Python is in the [Python examples with ResNet50](https://xilinx.github.io/inference-server/0.4.0/example_resnet50_python.html) and the corresponding working [Python scripts in the repository](https://github.com/Xilinx/inference-server/tree/main/examples/resnet50).


An outline of the steps is provided here, where it is assumed you have started the server and loaded some models that you want to use for inference. In general, the process to make a request has the following steps:

1. Make a client.
2. Request a worker from the server.
3. Prepare a request.
4. Send the request.
5. Check the response.

You can make an `HttpClient` or a `GrpcClient` depending on which protocol you want to use. As part of the constructor, you provide the address for the server that the client is supposed to use.

The structure of a request for the inference server is based on [KServe's v2 inference API](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference). You construct `InferenceRequestInput` objects and add the name, shape, datatype, and data associated with them and add them to an `InferenceRequest` object. Depending on the model, you may need to pre-process the data or have it in specific datatypes or formats.

Use the client's `modelInfer` method to make the inference request by providing the constructed request and the endpoint for the model you're using.

To check the data, you can analyze the returned `InferenceResponse` object to see the returned output tensors. Depending on the model, you might need to post-process the raw data to extract meaningful information from it.

This could look something like the following:

```python

# before making an inference, you need two things:
#   - the address of the server
#   - the endpoint that was returned from loading the worker

# using an HTTP address that's running locally
server_addr = "http:127.0.0.1:8998"
# this endpoint is just a string returned from the server that 
# will uniquely identify your worker when you load it
# endpoint = client.modelLoad(<name>, <parameters>)
endpoint = "xmodel"

# import the Python library
import amdinfer

# create the client and make sure everything is ready
client = amdinfer.HttpClient(server_addr)
amdinfer.waitUntilServerReady(client)
assert client.modelReady(endpoint)

# construct a request
request = amdinfer.InferenceRequest()
input_0 = amdinfer.InferenceRequestInput()
input_0.name = "a_name"
input_0.datatype = amdinfer.DataType.INT64
input_0.shape = [2, 3]
# data should be flattened
input_0.setInt64Data([0, 1, 2, 3, 4, 5])
request.addInputTensor(input_0)

# make the inference
response = client.modelInfer(endpoint, request)

# analyze the response
assert not response.isError()
outputs = response.getOutputs()
assert len(outputs) = 1
data = (outputs[0]).getInt64Data()
print(data)
```

<hr/>

[< Previous](/docs/4_deploy_your_own_model/deploy_model/deployingmodel.md) | [Next >](/docs/5_debugging_and_profiling/debugging_and_profiling.md)

<hr/>

# License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.
