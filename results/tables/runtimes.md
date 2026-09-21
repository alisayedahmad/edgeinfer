# runtime comparison

ds-cnn keyword spotter, speech commands v2, batch 1.

| runtime     | precision    | accuracy % | p50 ms  | p90 ms  | size kb | peak ram kb    | kernels |
|-------------|--------------|------------|---------|---------|---------|----------------|---------|
| pytorch     | fp32         | 93.98      | 4.720   | 6.191   | 549.1   | n/a            | 0       |
| onnxruntime | fp32         | 93.98      | 1.627   | 2.398   | 568.3   | n/a            | 13      |
| onnxruntime | fp32-unopt   | 93.98      | 2.684   | 3.800   | 568.3   | n/a            | 30      |
| onnxruntime | int8         | 93.98      | 0.463   | 0.560   | 182.3   | n/a            | 16      |
| tflite      | fp32         | 93.98      | 2.894   | 3.461   | 547.7   | 6088 (process) | 0       |
| tflite      | int8         | 93.77      | 122.544 | 263.278 | 146.8   | 4748 (process) | 0       |
| c_engine    | fp32         | 93.98      | 19.366  | 20.484  | 543.0   | 168 (planner)  | 9       |
| c_engine    | fp32-unfused | 93.98      | 40.777  | 41.895  | 549.1   | 168 (planner)  | 0       |
| c_engine    | int8         | 94.00      | 29.077  | 30.424  | 148.1   | 42 (planner)   | 9       |

## time per operator (ms)

| runtime                | conv2d        | depthwise_conv | pointwise_conv | batch_norm    | relu          | pool          | dense         | other         |
|------------------------|---------------|----------------|----------------|---------------|---------------|---------------|---------------|---------------|
| pytorch fp32           | -             | -              | -              | -             | -             | -             | -             | -             |
| onnxruntime fp32       | 0.155 ( 8.5%) | 0.457 (24.9%)  | 1.080 (58.8%)  | -             | -             | 0.043 ( 2.3%) | 0.037 ( 2.0%) | 0.064 ( 3.5%) |
| onnxruntime fp32-unopt | 0.104 ( 3.1%) | 1.572 (47.2%)  | 0.890 (26.8%)  | 0.459 (13.8%) | 0.232 ( 7.0%) | 0.030 ( 0.9%) | 0.026 ( 0.8%) | 0.014 ( 0.4%) |
| onnxruntime int8       | 0.054 ( 8.8%) | 0.191 (31.2%)  | 0.302 (49.5%)  | -             | -             | 0.013 ( 2.1%) | 0.015 ( 2.4%) | 0.037 ( 6.1%) |
| tflite fp32            | -             | -              | -              | -             | -             | -             | -             | -             |
| tflite int8            | -             | -              | -              | -             | -             | -             | -             | -             |
| c_engine fp32          | 2.329 (11.3%) | 0.636 ( 3.1%)  | 17.647 (85.5%) | -             | -             | 0.019 ( 0.1%) | 0.009 ( 0.0%) | -             |
| c_engine fp32-unfused  | 3.492 ( 9.5%) | 0.978 ( 2.7%)  | 31.523 (86.2%) | 0.317 ( 0.9%) | 0.225 ( 0.6%) | 0.023 ( 0.1%) | 0.014 ( 0.0%) | -             |
| c_engine int8          | 4.815 (16.4%) | 2.729 ( 9.3%)  | 21.801 (74.2%) | -             | -             | 0.028 ( 0.1%) | 0.010 ( 0.0%) | -             |

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
