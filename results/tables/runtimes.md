# runtime comparison

ds-cnn keyword spotter, speech commands v2, batch 1.

| runtime     | precision    | accuracy % | p50 ms  | p90 ms  | size kb | peak ram kb   | kernels run |
|-------------|--------------|------------|---------|---------|---------|---------------|-------------|
| pytorch     | fp32         | 93.98      | 1.964   | 2.136   | 549.1   | 6504 (rss)    | -           |
| onnxruntime | fp32         | 93.98      | 0.468   | 0.536   | 568.3   | 7328 (rss)    | 13          |
| onnxruntime | fp32-unopt   | 93.98      | 1.206   | 1.238   | 568.3   | 6500 (rss)    | 30          |
| onnxruntime | int8         | 93.98      | 0.450   | 0.512   | 182.3   | 6344 (rss)    | 16          |
| tflite      | fp32         | 93.98      | 2.864   | 2.926   | 547.7   | 4660 (rss)    | 13          |
| tflite      | int8         | 93.77      | 123.465 | 126.270 | 146.8   | 4176 (rss)    | 13          |
| c_engine    | fp32         | 93.98      | 16.921  | 17.067  | 543.0   | 168 (planner) | 11          |
| c_engine    | fp32-unfused | 93.98      | 17.349  | 17.695  | 549.1   | 168 (planner) | 29          |
| c_engine    | int8         | 94.00      | 6.059   | 6.115   | 148.1   | 42 (planner)  | 11          |

## time per operator (ms)

| runtime                | conv2d        | depthwise_conv | pointwise_conv | batch_norm    | relu          | pool          | dense         | other         |
|------------------------|---------------|----------------|----------------|---------------|---------------|---------------|---------------|---------------|
| onnxruntime fp32       | 0.037 ( 7.0%) | 0.114 (21.2%)  | 0.352 (65.7%)  | -             | -             | 0.010 ( 1.9%) | 0.008 ( 1.4%) | 0.015 ( 2.8%) |
| onnxruntime fp32-unopt | 0.040 ( 2.9%) | 0.694 (49.6%)  | 0.384 (27.4%)  | 0.168 (12.0%) | 0.087 ( 6.2%) | 0.012 ( 0.9%) | 0.008 ( 0.6%) | 0.006 ( 0.4%) |
| onnxruntime int8       | 0.052 ( 9.3%) | 0.174 (30.9%)  | 0.284 (50.6%)  | -             | -             | 0.010 ( 1.9%) | 0.012 ( 2.1%) | 0.030 ( 5.3%) |
| c_engine fp32          | 2.789 (16.5%) | 0.136 ( 0.8%)  | 13.993 (82.7%) | -             | -             | 0.004 ( 0.0%) | 0.006 ( 0.0%) | -             |
| c_engine fp32-unfused  | 3.220 (18.5%) | 0.109 ( 0.6%)  | 13.993 (80.4%) | 0.042 ( 0.2%) | 0.023 ( 0.1%) | 0.002 ( 0.0%) | 0.006 ( 0.0%) | -             |
| c_engine int8          | 1.132 (18.7%) | 0.856 (14.1%)  | 4.066 (67.0%)  | -             | -             | 0.011 ( 0.2%) | 0.002 ( 0.0%) | -             |

no per-operator breakdown for pytorch, tflite on this machine.

## measured on

- Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz, 8 cores, windows 11, python 3.13.9, commit `3d952b3`: pytorch fp32, onnxruntime fp32, onnxruntime fp32-unopt, onnxruntime int8, tflite fp32, tflite int8, c_engine fp32, c_engine fp32-unfused, c_engine int8

## cortex-m4 target

| cortex-m4          | used            | budget          |      |
|--------------------|-----------------|-----------------|------|
| flash              | 740.2 kb        | 1024 kb         | fits |
| ram (bss + stack)  | 184.2 kb        | 256 kb          | fits |
| arena int8         | 42.0 kb greedy  | 189.6 kb naive  |      |
| arena fp32         | 168.0 kb greedy | 758.6 kb naive  |      |
| arena fp32-unfused | 168.0 kb greedy | 2270.3 kb naive |      |

| mode | mfcc (m instructions) | inference (m instructions) | predicted   |
|------|-----------------------|----------------------------|-------------|
| int8 | 10.24                 | 50.98                      | 6 (label 6) |
| fp32 | 10.24                 | 1098.07                    | 6 (label 6) |
