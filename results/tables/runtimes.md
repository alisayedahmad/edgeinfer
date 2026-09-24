# runtime comparison

ds-cnn keyword spotter, speech commands v2, batch 1.

| runtime     | precision    | accuracy % | p50 ms  | p90 ms  | size kb | peak ram kb   | kernels run |
|-------------|--------------|------------|---------|---------|---------|---------------|-------------|
| pytorch     | fp32         | 93.98      | 2.082   | 2.829   | 549.1   | 6432 (rss)    | -           |
| onnxruntime | fp32         | 93.98      | 0.467   | 0.560   | 568.3   | 7296 (rss)    | 13          |
| onnxruntime | fp32-unopt   | 93.98      | 1.202   | 1.417   | 568.3   | 6396 (rss)    | 30          |
| onnxruntime | int8         | 93.98      | 0.451   | 0.496   | 182.3   | 6648 (rss)    | 16          |
| tflite      | fp32         | 93.98      | 2.888   | 3.335   | 547.7   | 4484 (rss)    | 13          |
| tflite      | int8         | 93.77      | 123.163 | 124.801 | 146.8   | 4036 (rss)    | 13          |
| c_engine    | fp32         | 93.98      | 17.226  | 17.649  | 543.0   | 168 (planner) | 11          |
| c_engine    | fp32-unfused | 93.98      | 17.613  | 18.267  | 549.1   | 168 (planner) | 29          |
| c_engine    | int8         | 94.00      | 8.175   | 8.409   | 148.1   | 42 (planner)  | 11          |

## time per operator (ms)

| runtime                | conv2d        | depthwise_conv | pointwise_conv | batch_norm    | relu          | pool          | dense         | other         |
|------------------------|---------------|----------------|----------------|---------------|---------------|---------------|---------------|---------------|
| onnxruntime fp32       | 0.085 ( 6.9%) | 0.263 (21.1%)  | 0.809 (64.9%)  | -             | -             | 0.025 ( 2.0%) | 0.023 ( 1.8%) | 0.041 ( 3.3%) |
| onnxruntime fp32-unopt | 0.086 ( 2.7%) | 1.540 (48.6%)  | 0.871 (27.5%)  | 0.408 (12.9%) | 0.207 ( 6.5%) | 0.026 ( 0.8%) | 0.021 ( 0.7%) | 0.011 ( 0.3%) |
| onnxruntime int8       | 0.126 ( 9.6%) | 0.396 (30.1%)  | 0.665 (50.6%)  | -             | -             | 0.026 ( 2.0%) | 0.031 ( 2.4%) | 0.070 ( 5.4%) |
| c_engine fp32          | 2.911 (16.8%) | 0.138 ( 0.8%)  | 14.238 (82.3%) | -             | -             | 0.006 ( 0.0%) | 0.007 ( 0.0%) | -             |
| c_engine fp32-unfused  | 3.282 (18.5%) | 0.108 ( 0.6%)  | 14.230 (80.3%) | 0.059 ( 0.3%) | 0.026 ( 0.1%) | 0.002 ( 0.0%) | 0.007 ( 0.0%) | -             |
| c_engine int8          | 3.142 (38.3%) | 0.911 (11.1%)  | 4.140 (50.4%)  | -             | -             | 0.013 ( 0.2%) | 0.002 ( 0.0%) | -             |

no per-operator breakdown for pytorch, tflite on this machine.

## measured on

- Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz, 8 cores, windows 11, python 3.13.9, commit `894349b`: pytorch fp32, onnxruntime fp32, onnxruntime fp32-unopt, onnxruntime int8, tflite fp32, tflite int8, c_engine fp32, c_engine fp32-unfused, c_engine int8

## cortex-m4 target

| cortex-m4          | used            | budget          |      |
|--------------------|-----------------|-----------------|------|
| flash              | 738.7 kb        | 1024 kb         | fits |
| ram (bss + stack)  | 184.2 kb        | 256 kb          | fits |
| arena int8         | 42.0 kb greedy  | 189.6 kb naive  |      |
| arena fp32         | 168.0 kb greedy | 758.6 kb naive  |      |
| arena fp32-unfused | 168.0 kb greedy | 2270.3 kb naive |      |

| mode | mfcc (m instructions) | inference (m instructions) | predicted   |
|------|-----------------------|----------------------------|-------------|
| int8 | 10.24                 | 132.80                     | 6 (label 6) |
| fp32 | 10.24                 | 1098.07                    | 6 (label 6) |
