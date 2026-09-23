# int8 error by layer

error relative to each layer's own fp32 range.

## c engine

| layer | max abs | max % of range | rms % of range |
|-------|---------|----------------|----------------|
| conv1 | 0.1881  | 1.914          | 0.168          |
| dw1   | 0.4198  | 2.525          | 0.176          |
| pw1   | 0.2924  | 3.201          | 0.326          |
| dw2   | 0.7092  | 4.459          | 0.245          |
| pw2   | 0.5193  | 4.737          | 0.331          |
| dw3   | 0.7105  | 5.417          | 0.326          |
| pw3   | 0.4411  | 4.690          | 0.225          |
| dw4   | 2.3239  | 6.546          | 0.140          |
| pw4   | 1.0257  | 8.782          | 0.436          |
| pool  | 0.0713  | 7.178          | 1.982          |
| fc    | 0.3283  | 3.330          | 0.795          |

## tflite

| layer                            | max abs | max % of range | rms % of range |
|----------------------------------|---------|----------------|----------------|
| dw1_input_nhwc                   | 0.1593  | 1.621          | 0.198          |
| pw1_input_nhwc                   | 0.4753  | 2.858          | 0.203          |
| dw2_input_nhwc                   | 0.3412  | 3.735          | 0.371          |
| pw2_input_nhwc                   | 0.7784  | 4.895          | 0.296          |
| dw3_input_nhwc                   | 0.6148  | 5.608          | 0.398          |
| pw3_input_nhwc                   | 0.7663  | 5.843          | 0.408          |
| dw4_input_nhwc                   | 0.5387  | 5.727          | 0.286          |
| pw4_input_nhwc                   | 2.7483  | 7.742          | 0.241          |
| features/pw4/pw4.2/Relu_output_0 | 10.1875 | 87.219         | 4.261          |
| pool/GlobalAveragePool_output_0  | 0.6195  | 62.379         | 13.966         |
| logits                           | 2.7233  | 27.624         | 9.170          |

