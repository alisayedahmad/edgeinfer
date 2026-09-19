"""tensorrt inference with per-layer profiling.

    python -m runtime.tensorrt_runner --precision fp32|fp16|int8

engines are built by train/export.py. precision comes from the onnx tensor
types, since tensorrt 11 networks are strongly typed and the fp16/int8
builder flags are gone. per-layer timings come from an IProfiler, and the
fused layer names from the engine inspector json written at build time.

device buffers are plain torch cuda tensors, so there is no pycuda here.
"""
import argparse
import json

import numpy as np
import torch

from runtime import profile

TORCH_DTYPE = {"FLOAT": torch.float32, "HALF": torch.float16, "INT8": torch.int8,
               "INT32": torch.int32, "BOOL": torch.bool}


def load(precision):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    path = profile.ARTIFACTS / "trt" / f"ds_cnn_{precision}.engine"
    engine = trt.Runtime(logger).deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"could not deserialize {path}, rebuild it on this gpu")
    return trt, engine, path


def io_tensors(trt, engine, context, batch):
    """device buffers for every engine tensor, shapes set for this batch."""
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = (batch, *engine.get_tensor_shape(name)[1:])
            context.set_input_shape(name, shape)
        else:
            shape = tuple(context.get_tensor_shape(name))
        dtype = TORCH_DTYPE[engine.get_tensor_dtype(name).name]
        buffers[name] = torch.empty(tuple(shape), dtype=dtype, device="cuda")
        context.set_tensor_address(name, buffers[name].data_ptr())
    return buffers


def infer(context, stream, buffers, names, x):
    buffers[names[0]][: len(x)].copy_(torch.from_numpy(x).to("cuda", non_blocking=True))
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return buffers[names[1]][: len(x)].float().cpu().numpy()


def run(precision, runs, batch):
    trt, engine, path = load(precision)
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    x, y = profile.eval_set()

    buffers = io_tensors(trt, engine, context, batch)
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    logits = []
    for i in range(0, len(x), batch):
        chunk = x[i:i + batch]
        if len(chunk) != batch:
            buffers = io_tensors(trt, engine, context, len(chunk))
        logits.append(infer(context, stream, buffers, names, chunk))
    logits = np.concatenate(logits)

    buffers = io_tensors(trt, engine, context, 1)
    one = x[:1]

    class Times(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.ms = {}

        def report_layer_time(self, layer_name, ms):
            self.ms[layer_name] = self.ms.get(layer_name, 0.0) + ms

    latency = profile.latency(lambda: infer(context, stream, buffers, names, one), runs)

    # profiling serializes every layer, so time it separately from latency
    times = Times()
    context.profiler = times
    for _ in range(50):
        infer(context, stream, buffers, names, one)
    context.profiler = None

    layers = json.loads(path.with_suffix(".layers.json").read_text())
    layers = layers.get("Layers", layers) if isinstance(layers, dict) else layers
    device_bytes = getattr(engine, "device_memory_size_v2", None) or engine.device_memory_size
    return {
        "runtime": "tensorrt", "precision": precision,
        "accuracy": profile.accuracy(logits, y),
        "latency_ms": latency,
        "model_size_kb": path.stat().st_size / 1024,
        "peak_ram_kb": device_bytes / 1024, "peak_ram_source": "engine device memory",
        "ops": [{"name": name, "op": profile.canonical_op(name), "ms": ms / 50, "bytes": 0}
                for name, ms in times.ms.items()],
        "fused": [layer if isinstance(layer, str) else layer.get("Name", "") for layer in layers],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="tensorrt inference and profiling")
    p.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    profile.save(run(args.precision, args.runs, args.batch))
