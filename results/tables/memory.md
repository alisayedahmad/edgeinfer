# memory

arena sizes from the c engine planner.

| mode         | naive kb | planned kb | saved | buffers | largest tensor kb |
|--------------|----------|------------|-------|---------|-------------------|
| fp32         | 758.6    | 168.0      | 4.5x  | 2       | 84.0              |
| fp32-unfused | 2270.3   | 168.0      | 13.5x | 2       | 84.0              |
| int8         | 189.6    | 42.0       | 4.5x  | 2       | 21.0              |

| runtime               | peak ram kb | how it was measured          |
|-----------------------|-------------|------------------------------|
| c_engine fp32-unfused | 168         | planner arena                |
| c_engine fp32         | 168         | planner arena                |
| c_engine int8         | 42          | planner arena                |
| tflite fp32           | 6088        | process rss high-water delta |
| tflite int8           | 4748        | process rss high-water delta |
