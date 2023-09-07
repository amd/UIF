<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Unified Inference Frontend (UIF) 1.2 User Guide </h1>
    </td>
 </tr>
 <tr>
 <td align="center"><h1>Step 5.0: Debugging and Profiling</h1>
 </td>
 </tr>
</table>

# Table of Contents

- [5.1: Debug on GPU](#51-debug-on-gpu)
- [5.2: Debug on CPU](#52-debug-on-cpu)
- [5.3: Debug on FPGA](#53-debug-on-fpga)
  

  _Click [here](/README.md#implementing-uif-11) to go back to the UIF User Guide home page._


# 5.1: Debug on GPU

## 5.1.1: ROCDebugger

The purpose of a debugger such as ROCGDB is to allow you to see what is going on “inside” another program while it executes—or what another program was doing at the moment it crashed.

ROCGDB can do four main kinds of things to help you catch bugs in the act:

- Start your program, specifying anything that might affect its behavior.
- Make your program stop on specified conditions.
- Examine what has happened, when your program has stopped.
- Change things in your program, so you can experiment with correcting the effects of one bug and go on to learn about another.

For more information about ROCDebugger, refer to the [ROCDebugger User Guide](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/ROCgdb/gdb/doc/gdb/index.html).

## 5.1.2: ROCProfiler

This document describes rocprof as the AMD heterogeneous system GPU/CPU profiling and tracing tool. rocprof is a command line tool implemented on top of the ROCm™ platform’s profiling libraries – rocProfiler and rocTracer. The input to rocprof is an XML or a text file that contains counters list or trace parameters. The output is profiling data and statistics in various formats such as text, CSV, and JSON traces.

The following user requirements can be fulfilled using rocprof:

### 5.1.2.1: Counters and Metric Collection
To collect counters and metrics such as number of VMEM read/write instructions issued, number of SALU instructions issued, and other details, use rocprof with profiling options.

For more details, refer to the chapter on Counter and Metric Collection in the [ROCm Profiling Tools User Guide](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/profiler_home_page.html).

### 5.1.2.2: Application Tracing

To retrieve kernel-level traces such as workgroup size, HIP/HSA calls, and so on, use rocprof with tracing options such as hsa-trace, hip-trace, sys-trace, and roctx-trace.

To demonstrate the usage of rocprof with various options, the [ROCm Profiling Tools User Guide](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/profiler_home_page.html) refers to the MatrixTranspose application as an example.

# 5.2: Debug on CPU

This section describes some common issues you could encounter with the CPU inference software stack and how to get around them.

## 5.2.1: Frequently Encountered Issues

1. The ZenDNN Model Converter throws the error: `This model is not supported for optimization`.

    Some models might not be optimized with ZenDNN. Refer to the list of supported models for [ZenDNN Model Converter](/docs/4_deploy_your_own_model/deploy_model/deployingmodel.md#2-quantizing-a-deep-learning-model) for details.

2. The ZenDNN Model Converter throws the error: `Mixed input attributes or qint8 input to Concat is not supported` or `VitisAIConv2DWithSum does not support int input and float output`.

    If there is a potential attribute mismatch between successive nodes, or an unsupported attribute combination for an operation is found during the conversion, this error will come up.

3. Model throws the error: `ZENDNN_BLOCKED_FORMAT not supported for this model, please use another data format`.

    This means that the model does not have channels as a multiple of 8 and therefore `ZENDNN_CONV_ALGO=3` cannot be used for this model. Use any other value for `ZENDNN_CONV_ALGO`. For more information, refer to the [TensorFlow+ZenDNN documentation](https://www.amd.com/en/developer/zendnn.html).

4. My model runs slow on ZenDNN.

    Check the performance/tuning guidelines provided in the [documentation](https://www.amd.com/en/developer/zendnn.html) for the best environment variables to use. Additionally, each UIF ZenDNN model will be provided with a recommended setting for best performance.

## 5.2.2. Debugging/Profiling Using ZenDNN Execution Logs

ZenDNN provides execution logs of each primitive in the model. These execution logs can be used for debugging or profiling the model execution.

Logging is disabled in the ZenDNN library by default. It can be enabled using the environment variable `ZENDNN_LOG_OPTS` before running any models. Refer to the [ZenDNN User Guide](https://www.amd.com/en/developer/zendnn.html) for more information on how to enable logs.

If you encounter any issues, email zendnnsupport@amd.com with your model details and logs.

## 5.2.3. Detailed Profiling with AMD Zen Studio

System-level profiling for model execution can be done with AMD μProf from AMD Zen Studio. For more information, refer to [AMD μProf](https://www.amd.com/en/developer/uprof.html).

# 5.3: Debug on FPGA

This section describes the utility tools available in UIF 1.2 for DPU execution debugging, performance profiling, DPU runtime mode manipulation, and DPU configuration file generation. With these tools, you can conduct DPU debugging and performance profiling independently.

## 5.3.1: Profiling the Model

UIF Profiler is an application-level tool that profiles and visualizes AI applications based on VART. For an AI application, there are components that run on the hardware; for example, neural network computation usually runs on the DPU, and there are components that run on a CPU as a function that is implemented by C/C++ code-like image pre-processing. This tool helps you to put the running status of all these different components together.

For more information on profiling the model in UIF, see [Profiling the Model](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Profiling-the-Model).

## 5.3.2: Inspecting the Model

UIF Inspector inspects a float model and show partition results for a given DPU target architecture, together with some indications on why the layers are not mapped to DPU. Without a target, you can only show some general, target-independent inspection results. Assign a target to get more detailed inspect results for it.

For more information on inspecting the model in UIF, see [Inspecting the Float Model](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Inspecting-the-Float-Model) in the Vitis™ AI documentation.


<hr/>

[< Previous](/docs/4_deploy_your_own_model/serve_model/servingmodelwithinferenceserver.md)

<hr/>

# License

UIF is licensed under [Apache License Version 2.0](/LICENSE). Refer to the [LICENSE](/LICENSE) file for the full license text and copyright notice.

# Technical Support

Contact uif_support@amd.com for questions, issues, and feedback on UIF.

Submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/UIF/issues) page.

