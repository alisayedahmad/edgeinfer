# fusion

## what the exporter already folded

| export                      | nodes | conv | batchnorm | relu | rest                                                    |
|-----------------------------|-------|------|-----------|------|---------------------------------------------------------|
| torchscript (ds_cnn.onnx)   | 30    | 9    | 9         | 9    | GlobalAveragePool x1, Flatten x1, Gemm x1               |
| dynamo (ds_cnn_dynamo.onnx) | 23    | 9    | 0         | 9    | Shape x1, ReduceMean x1, Concat x1, Reshape x1, Gemm x1 |

## what each runtime runs

| runtime                | nodes in | kernels run | merged names | example                       |
|------------------------|----------|-------------|--------------|-------------------------------|
| c_engine fp32-unfused  | -        | 29          | -            |                               |
| c_engine fp32          | -        | 11          | 9            | conv1 + conv1_bn + conv1_relu |
| c_engine int8          | -        | 11          | 9            | conv1 + conv1_bn + conv1_relu |
| onnxruntime fp32-unopt | 30       | 30          | -            |                               |
| onnxruntime fp32       | 30       | 13          | -            |                               |
| onnxruntime int8       | 58       | 16          | -            |                               |
| pytorch fp32           | -        | -           | -            |                               |
| tflite fp32            | -        | 13          | -            |                               |
| tflite int8            | -        | 13          | -            |                               |

## fusion impact

| configuration                      | p50 ms | peak ram kb | kernels |
|------------------------------------|--------|-------------|---------|
| c engine, unfused (conv, bn, relu) | 17.613 | 168         | 29      |
| c engine, manual fusion            | 17.226 | 168         | 11      |
| onnx runtime, optimizations off    | 1.202  | 6396        | 30      |
| onnx runtime, optimizations on     | 0.467  | 7296        | 13      |

| numerical difference (max abs logit) | value     |
|--------------------------------------|-----------|
| c engine fused vs unfused            | 8.345e-06 |
| onnx runtime optimized vs not        | 2.623e-06 |

fusion saves two passes over the activations per layer, so on a desktop cpu with
small tensors the win can fall inside run-to-run noise. the deterministic
evidence is the operator breakdown in runtimes.md: the batch_norm and relu rows
of an unfused run are exactly what fusing removes.
