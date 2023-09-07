<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.2 User Guide </h1>
    </td>
 </tr>
 <tr>
 <td align="center"><h1> Step 3.2: Run an Example with the Inference Server</h1>
 </td>
 </tr>
</table>

This example walks you through running ResNet50 examples with the Inference Server using the development and deployment containers based on the [developer quickstart](https://xilinx.github.io/inference-server/0.4.0/quickstart_development.html) guide.
An easier example using just the deployment container is in the [quickstart](https://xilinx.github.io/inference-server/0.4.0/quickstart.html).
The full example files described here are available in the Inference Server [repository](https://github.com/Xilinx/inference-server/tree/main/examples/resnet50).
The repository has examples for three backends: CPU (with ZenDNN), GPU (with MIGraphX), and FPGA (with Vitis&trade; AI).
This example uses the GPU backend but all backends behave similarly.
For the scripts used in this example, you can always use `--help` to see the available options.

# Table of Contents
- [3.2.1: Get the Code and Build the Image](#321-get-the-code-and-build-the-image)
- [3.2.2: Start the Container](#322-start-the-container)
- [3.2.3: ResNet50 Example](#323-resnet50-example)
- [3.2.4: Extending the Example](#324-extending-the-example)

_Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._

# 3.2.1: Get the Code and Build the Image

1. Clone the Inference Server repository:

```bash
git clone https://github.com/Xilinx/inference-server.git
cd inference-server
```

2. Build a development Docker® image for the Inference Server based on the [installation instructions](/docs/1_installation/installation.md#142-build-an-inference-server-docker-image).
Enable one or more platforms using the appropriate flags depending on which example(s) you want to run.
Because this example is only using the GPU backend, it is enabled with the `--migraphx` flag.
The optional `--suffix` flag modifies the name of the image.

```bash
python3 docker/generate.py
./amdinfer dockerize --migraphx --suffix="-migraphx"
```

This builds the development image with the name `$(whoami)/amdinfer-dev-migraphx:latest` on your host.
The development image only contains the dependencies for the Inference Server but not the source code.
When you start the container, mount this directory inside so you can compile it.

3. Get the example models and test data.
You can pass the appropriate flag(s) for your backend to get the relevant files or use the `--all` flag to get everything.
This command downloads these files and save them so they can be used for inference.
You need `git-lfs` to get test data.
You can install it on your host or run this command from inside the development container that already has it installed.

```bash
./amdinfer get --migraphx
```

# 3.2.2: Start the Container

You can start the deployment container with the running server with:

```bash
docker pull amdih/serve:uif1.2_migraphx_amdinfer_0.4.0
docker run -d --device /dev/kfd --device /dev/dri --volume $(pwd):/workspace/amdinfer:rw --network=host amdih/serve:uif1.2_migraphx_amdinfer_0.4.0
```

This starts the server in detached mode, mount the GPU into the container as well as the current inference server repository containing the models.
By default, the server uses port 8998 for HTTP requests and it shares the host network for easier networking to remote clients.

You can confirm the server is ready by using `curl` on the host to see if the command below succeeds.
If you are going to make requests from a different host, replace the localhost address with the IP address of the host where the server is running.

```bash
curl http://127.0.0.1:8998/v2/health/ready
```

You can start the development container with:

```bash
docker run -it --rm --network=host --volume $(pwd):/workspace/amdinfer:rw --workdir /workspace/amdinfer $(whoami)/amdinfer-dev-migraphx:latest
```

By using `--network=host`, the development container can more easily connect to the server running in the deployment container.

## 3.2.3: ResNet50 Example

The remainder of the commands in this example are run in the terminal inside the development container.
Before you can run the example, you need to build the Python library.

```bash
amdinfer build
```

You can run the example with:

```bash
python ./examples/resnet50/migraphx.py --ip <ip address> --http-port <port>
```

If the server is running on a different host, pass the IP address to that host or use 127.0.0.1 if the server is running on the same host where you are making the request.
By passing the correct IP and port of the running server, this example script connects to the running server in the deployment container.

This example script carries out the following steps:

1. Starts the server if it is not already started. It prints a message if it does this. By passing the right address for it to connect to, the script does not attempt to start a server itself.
2. Loads a MIGraphX worker to handle the incoming inference request. The worker opens and compiles a ResNet50 ONNX model with MIGraphX.
3. Opens and preprocesses an image of a dog for ResNet50 and uses it to make an inference request.
4. Sends the request over HTTP REST to the server. The server responds with the output of the model.
5. Takes the response and postprocesses it to extract the top five classifications for the image.
7. Prints the labels associated with the top five classifications.

## 3.2.4: Extending the Example

The example script has a number of flags you can use to change its behavior.
For example, you can use `--image` to pass a path to your own image to the ResNet50 model.

The other ResNet50 examples using different backends work similarly but use different workers to implement the functionality.
Some of the other examples also demonstrate using different communication protocols like gRPC to communicate with the server instead of HTTP REST.
Changing protocols is easy: just change the client you are using to make inferences.
To run other examples, you need a compatible Docker image.
You can build a new container with another backend or enable multiple backends in one by passing in multiple flags.
You can see more information about what the example script does in the [Python examples](https://xilinx.github.io/inference-server/0.4.0/example_resnet50_python.html) in the Inference Server documentation.

There are also C++ versions of these examples in the repository.
You can see more information about the [C++ examples](https://xilinx.github.io/inference-server/0.4.0/example_resnet50_cpp.html) in the Inference Server documentation.

<hr/>

[< Previous](/docs/3_run_example/runexample-script.md) | [Next >](/docs/3_run_example/runexample-migraphx.md)

<hr/>

# License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.
