# memory

arena sizes from the c engine planner.

| mode         | naive kb | planned kb | saved | buffers | largest tensor kb |
|--------------|----------|------------|-------|---------|-------------------|
| fp32         | 758.6    | 168.0      | 4.5x  | 2       | 84.0              |
| fp32-unfused | 2270.3   | 168.0      | 13.5x | 2       | 84.0              |
| int8         | 189.6    | 42.0       | 4.5x  | 2       | 21.0              |

| runtime                | peak ram kb | how it was measured                    |
|------------------------|-------------|----------------------------------------|
| c_engine fp32-unfused  | 168         | planner arena                          |
| c_engine fp32          | 168         | planner arena                          |
| c_engine int8          | 42          | planner arena                          |
| onnxruntime fp32-unopt | 6396        | rss for the runtime plus one inference |
| onnxruntime fp32       | 7296        | rss for the runtime plus one inference |
| onnxruntime int8       | 6648        | rss for the runtime plus one inference |
| pytorch fp32           | 6432        | rss for the runtime plus one inference |
| tflite fp32            | 4484        | rss for the runtime plus one inference |
| tflite int8            | 4036        | rss for the runtime plus one inference |
