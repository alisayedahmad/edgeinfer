"""export a trained checkpoint to every runtime.

    python -m train.export --targets onnx tflite tensorrt c

all int8 paths share one calibration set (1000 clean training clips) and
every runtime is scored on one evaluation set (the full test split), both
written to artifacts/ so the runners never touch the raw dataset.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from data import speech_commands as sc
from train import quant
from train.ds_cnn import N_FRAMES, N_MFCC, layer_names, load
from train.train import features

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
WEIGHTS = REPO / "c_engine" / "weights"
TARGETS = ["onnx", "tflite", "tensorrt", "c"]


def feature_sets(ckpt, n_calib, synthetic, seed=0):
    """(calibration x, test x, test y, clip), x normalized (n, 1, 49, 10) float32.

    clip is one raw 1 s int16 recording with its label, for the cortex-m4
    demo. synthetic swaps in random features and a chirp so the whole export
    path runs without the dataset, for ci and smoke tests.
    """
    rng = np.random.default_rng(seed)
    if synthetic:
        x = rng.standard_normal((n_calib + 512, 1, N_FRAMES, N_MFCC)).astype(np.float32)
        t = np.arange(sc.N_SAMPLES) / sc.SR
        chirp = np.round(0.5 * np.sin(2 * np.pi * (300 + 1200 * t) * t) * 32768)
        return x[:n_calib], x[n_calib:], rng.integers(0, ckpt["n_classes"], 512), (chirp.astype(np.int16), 0)
    mean, std = torch.as_tensor(ckpt["feat_mean"]), torch.as_tensor(ckpt["feat_std"])
    train_audio, _ = sc.load_split("train")
    pick = np.sort(rng.choice(len(train_audio), n_calib, replace=False))
    calib = features(train_audio[pick], mean, std, "cpu").numpy()
    test_audio, test_y = sc.load_split("test")
    clip = (test_audio[0], int(test_y[0]))
    return calib, features(test_audio, mean, std, "cpu").numpy(), test_y, clip


def export_onnx(model, out):
    """unfused graph: conv, batchnorm and relu stay separate nodes.

    the torchscript exporter with training=PRESERVE maps each module to one
    node. the dynamo exporter's optimizer folds bn into conv before any
    runtime sees the graph, which would hide each runtime's own fusion
    decisions (see export_onnx_dynamo). nodes are renamed to the layer
    names used everywhere else: conv1, conv1_bn, conv1_relu, dw1, ..., fc.
    """
    x = torch.zeros(1, 1, N_FRAMES, N_MFCC)
    torch.onnx.export(
        model, (x,), str(out), input_names=["mfcc"], output_names=["logits"],
        dynamic_axes={"mfcc": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False, training=torch.onnx.TrainingMode.PRESERVE, opset_version=17,
    )
    m = onnx.load(out)
    names, layer = iter(layer_names(model.blocks)), None
    suffix = {"BatchNormalization": "_bn", "Relu": "_relu"}
    fixed = {"GlobalAveragePool": "pool", "Flatten": "flatten", "Gemm": "fc"}
    for node in m.graph.node:
        if node.op_type == "Conv":
            layer = next(names)
            node.name = layer
        elif node.op_type in suffix:
            node.name = layer + suffix[node.op_type]
        else:
            node.name = fixed.get(node.op_type, node.name)
    onnx.checker.check_model(m)
    onnx.save(m, out)


def export_onnx_dynamo(model, out):
    # default torch.onnx path, kept to show its optimizer already folds bn
    x = torch.zeros(1, 1, N_FRAMES, N_MFCC)
    torch.onnx.export(
        model, (x,), str(out), input_names=["mfcc"], output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=4096)}},
        dynamo=True, external_data=False, verbose=False,
    )


def export_onnx_fp16(src, out):
    """fp16 copy of the graph for tensorrt, input and output stay fp32.

    tensorrt 11 removed the fp16 builder flag. every network is strongly
    typed, so precision has to come from the tensor types in the file.
    """
    m = onnx.load(src)
    g = m.graph
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name))
    x, y = g.input[0].name, g.output[0].name
    for node in g.node:
        node.input[:] = [f"{x}_fp16" if i == x else i for i in node.input]
        node.output[:] = [f"{y}_fp16" if o == y else o for o in node.output]
    g.node.insert(0, helper.make_node("Cast", [x], [f"{x}_fp16"], name="cast_in", to=TensorProto.FLOAT16))
    g.node.append(helper.make_node("Cast", [f"{y}_fp16"], [y], name="cast_out", to=TensorProto.FLOAT))
    del g.value_info[:]
    onnx.checker.check_model(m)
    onnx.save(m, out)


def export_onnx_int8(src, out, calib, for_tensorrt=False):
    """qdq int8 copy of the graph from onnxruntime's minmax calibration.

    for_tensorrt gives symmetric int8 activations and float biases, the
    form tensorrt's explicit quantization accepts (tensorrt 11 dropped the
    calibrator api). otherwise activations are uint8, the fast path for
    onnxruntime's x86 int8 kernels.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quant_pre_process, quantize_static,
    )

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.batches = iter(np.array_split(calib, max(1, len(calib) // 64)))

        def get_next(self):
            x = next(self.batches, None)
            return None if x is None else {"mfcc": x}

    pre = out.with_name(out.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre))
    symmetric = {"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False}
    quantize_static(
        str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QInt8 if for_tensorrt else QuantType.QUInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
        extra_options=symmetric if for_tensorrt else {},
    )
    pre.unlink()


def export_tflite(onnx_path, out_dir, calib):
    """fp32 tflite from onnx2tf, int8 from tflite's own calibration.

    onnx2tf rewrites the graph to nhwc and saves a savedmodel next to its
    fp32 .tflite. the int8 model comes from TFLiteConverter with a
    representative dataset, full integer so input and output are int8.
    """
    import onnx2tf
    import tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter

    saved = out_dir / "saved_model"
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path), output_folder_path=str(saved), batch_size=1,
        output_signaturedefs=True, copy_onnx_input_output_names_to_tflite=True, non_verbose=True,
    )
    fp32 = out_dir / "ds_cnn_fp32.tflite"
    shutil.copy(next(saved.glob("*_float32.tflite")), fp32)
    shape = Interpreter(model_path=str(fp32)).get_input_details()[0]["shape"]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([x.reshape(shape)] for x in calib)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    (out_dir / "ds_cnn_int8.tflite").write_bytes(converter.convert())


def build_engine(onnx_path, engine_path, max_batch=256):
    """build a tensorrt engine, precision comes from the onnx tensor types.

    also writes <engine>.layers.json from the engine inspector, which lists
    the layers tensorrt actually built, fused ones included.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # strongly typed is the only mode in trt 11, opt-in before that
    strongly_typed = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    flags = 0 if int(trt.__version__.split(".")[0]) >= 11 else strongly_typed
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    dims = (1, N_FRAMES, N_MFCC)
    profile.set_shape(network.get_input(0).name, (1, *dims), (1, *dims), (max_batch, *dims))
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"tensorrt build failed for {onnx_path}")
    engine_path.write_bytes(bytes(blob))
    engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
    info = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    engine_path.with_suffix(".layers.json").write_text(info)


def c_array(ctype, name, values, per_line=12):
    # 9 significant digits round-trip a float32 exactly
    v = np.asarray(values, dtype=np.float32 if ctype == "float" else None).ravel()
    items = [f"{x:.8e}f" for x in v.tolist()] if ctype == "float" else [str(x) for x in v.tolist()]
    rows = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return f"static const {ctype} {name}[{len(v)}] = {{\n    " + ",\n    ".join(rows) + "\n};\n"


def c_header(name, body):
    return (f"// generated by train/export.py, do not edit\n"
            f"#ifndef EI_{name}_H\n#define EI_{name}_H\n\n{body}\n#endif\n")


def write_mfcc_tables(out_dir=WEIGHTS):
    """fixed-point tables for c_engine/src/mfcc.c, derived from data/speech_commands.py.

    hann window and fft twiddles in q15, mel filterbank as per-band bin
    ranges with q21 uint16 weights, dct in q30, and log2(1 + i/256) in q16.
    """
    fb = sc.mel_filters()
    start = [int(np.nonzero(row)[0][0]) for row in fb]
    length = [int(np.nonzero(row)[0][-1]) - s + 1 for row, s in zip(fb, start)]
    weights = np.concatenate([fb[m, s:s + n] for m, (s, n) in enumerate(zip(start, length))])
    angle = 2.0 * np.pi * np.arange(sc.N_FFT // 2) / sc.N_FFT

    def q15(x):
        return np.clip(quant.round_away(x * 32768.0), -32768, 32767).astype(np.int64)

    body = "".join([
        "#include <stdint.h>\n\n",
        f"#define MFCC_N_FFT {sc.N_FFT}\n#define MFCC_HOP {sc.HOP}\n#define MFCC_FRAMES {N_FRAMES}\n",
        f"#define MFCC_BINS {sc.N_FFT // 2 + 1}\n#define MFCC_MELS {sc.N_MELS}\n#define MFCC_COEFFS {sc.N_MFCC}\n",
        f"#define MFCC_MEL_SHIFT 21\n",
        f"#define MFCC_LOG10_2_Q29 {int(quant.round_away(10.0 * np.log10(2.0) * (1 << 29)))}\n",
        f"#define MFCC_FLOOR_Q16 ({int(quant.round_away(10.0 * np.log10(sc.AMIN) * 65536))})\n\n",
        c_array("int16_t", "mfcc_hann", q15(sc.hann())),
        c_array("int16_t", "mfcc_cos", q15(np.cos(angle))),
        c_array("int16_t", "mfcc_sin", q15(np.sin(angle))),
        c_array("uint16_t", "mfcc_mel_start", start),
        c_array("uint8_t", "mfcc_mel_len", length),
        c_array("uint16_t", "mfcc_mel_w", quant.round_away(weights * (1 << 21)).astype(np.int64)),
        c_array("int32_t", "mfcc_dct", quant.round_away(sc.dct_matrix() * (1 << 30)).astype(np.int64)),
        c_array("int32_t", "mfcc_log2_lut",
                quant.round_away(np.log2(1.0 + np.arange(257) / 256.0) * 65536).astype(np.int64)),
    ])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mfcc_tables.h").write_text(c_header("MFCC_TABLES", body))


def write_clip_header(clip, word, out_dir=WEIGHTS):
    # one raw clip for the cortex-m4 demo, its fixed-point mfcc runs on target
    pcm, label = clip
    (out_dir / "clip.h").write_text(c_header("CLIP", "".join([
        f"#define CLIP_LABEL {label}\n", f'static const char clip_word[] = "{word}";\n\n',
        c_array("int16_t", "clip_pcm", pcm),
    ])))


def write_c_headers(model, ckpt, q, out_dir=WEIGHTS):
    """model.h, geometry.h and the weights_*.h headers for the c engine.

    model.h is plain defines, safe to include anywhere. the rest are static
    tables that only forward.c includes. weights_f32.h holds the bn-folded
    layers the fused path runs,
    weights_raw.h the separate conv and bn params for the unfused path,
    weights_int8.h the quantized layers plus the mfcc -> int8 input scaling.
    """
    layers = quant.float_layers(model)
    (out_dir / "model.h").write_text(c_header("MODEL", "".join([
        f"#define EI_IN_H {N_FRAMES}\n#define EI_IN_W {N_MFCC}\n",
        f"#define EI_WIDTH {model.width}\n#define EI_BLOCKS {model.blocks}\n",
        f"#define EI_LAYERS {len(layers)}\n#define EI_CLASSES {model.n_classes}\n",
    ])))
    geom = ", ".join("{%d, %d, %d, %d, %d, %d}" % (*f["kernel"], *f["stride"], *f["pad"])
                     for f in layers)
    names = ", ".join(f'"{f["name"]}"' for f in layers)
    (out_dir / "geometry.h").write_text(c_header("GEOMETRY", "".join([
        '#include "forward.h"\n\n',
        f"static const conv_params_t ei_geom[EI_LAYERS] = {{{geom}}};\n",
        f"static const char *const ei_layer_names[EI_LAYERS] = {{{names}}};\n",
    ])))

    fused, raw = [], []
    for f in layers:
        n = f["name"]
        fused += [c_array("float", f"{n}_w", f["wf"]), c_array("float", f"{n}_b", f["bf"])]
        raw += [c_array("float", f"{n}_raw_w", f["w"]),
                c_array("float", f"{n}_bn_scale", f["bn_scale"]),
                c_array("float", f"{n}_bn_shift", f["bn_shift"])]
    table = ", ".join(f"{{{f['name']}_w, {f['name']}_b}}" for f in layers)
    prelude = ['#include "forward.h"\n\n']
    (out_dir / "weights_f32.h").write_text(c_header("WEIGHTS_F32", "".join(prelude + fused + [
        c_array("float", "fc_w", model.fc.weight.detach().numpy()),
        c_array("float", "fc_b", model.fc.bias.detach().numpy()),
        c_array("float", "ei_feat_mean", ckpt["feat_mean"]),
        c_array("float", "ei_feat_std", ckpt["feat_std"]),
        f"\nstatic const layer_f32_t ei_f32_layers[EI_LAYERS] = {{{table}}};\n",
    ])))
    table = ", ".join(f"{{{f['name']}_raw_w, {f['name']}_bn_scale, {f['name']}_bn_shift}}" for f in layers)
    (out_dir / "weights_raw.h").write_text(c_header("WEIGHTS_RAW", "".join(prelude + raw + [
        f"\nstatic const layer_raw_t ei_raw_layers[EI_LAYERS] = {{{table}}};\n",
    ])))

    def int8_layer(p, n):
        arrays = [c_array("int8_t", f"{n}_wq", p["w_q"]), c_array("int32_t", f"{n}_bq", p["b_q"]),
                  c_array("int32_t", f"{n}_mult", p["m0"]), c_array("int8_t", f"{n}_shift", p["shift"])]
        entry = (f"{{{n}_wq, {n}_bq, {n}_mult, {n}_shift, {p['in_zp']}, {p['out_zp']}, "
                 f"{p['act_min']}, {p['act_max']}, {p['out_scale']:.8e}f}}")
        return arrays, entry

    body, entries = [], []
    for p in q["layers"]:
        arrays, entry = int8_layer(p, p["name"])
        body += arrays
        entries.append(entry)
    fc_arrays, fc_entry = int8_layer(q["fc"], "fc")
    # mfcc (q16 db) -> normalized -> int8 model input, one requant per coefficient
    mean, std = ckpt["feat_mean"].astype(np.float64), ckpt["feat_std"].astype(np.float64)
    mults = [quant.quantize_multiplier(1.0 / (s * q["in_scale"] * 65536.0)) for s in std]
    (out_dir / "weights_int8.h").write_text(c_header("WEIGHTS_INT8", "".join(prelude + body + fc_arrays + [
        f"\n#define EI_IN_SCALE {q['in_scale']:.8e}f\n#define EI_IN_ZP ({q['in_zp']})\n\n",
        f"static const layer_q_t ei_int8_layers[EI_LAYERS] = {{\n    " + ",\n    ".join(entries) + "\n};\n",
        f"static const layer_q_t ei_int8_fc = {fc_entry};\n\n",
        c_array("int32_t", "ei_mfcc_mean_q16", quant.round_away(mean * 65536.0).astype(np.int64)),
        c_array("int32_t", "ei_mfcc_mult", [m for m, _ in mults]),
        c_array("int8_t", "ei_mfcc_shift", [s for _, s in mults]),
    ])))


def main():
    p = argparse.ArgumentParser(description="export ds-cnn to onnx, tflite, tensorrt and c")
    p.add_argument("--ckpt", type=Path, default=ARTIFACTS / "ds_cnn.pt")
    p.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS + ["mfcc"])
    p.add_argument("--calib", type=int, default=1000, help="calibration clips shared by every int8 path")
    p.add_argument("--synthetic", action="store_true", help="random features instead of the dataset")
    args = p.parse_args()

    if "mfcc" in args.targets or "c" in args.targets:
        write_mfcc_tables()
    if args.targets == ["mfcc"]:
        return

    model, ckpt = load(args.ckpt)
    calib, test_x, test_y, clip = feature_sets(ckpt, args.calib, args.synthetic)
    ARTIFACTS.mkdir(exist_ok=True)
    np.save(ARTIFACTS / "calib_features.npy", calib)
    np.savez(ARTIFACTS / "test_features.npz", x=test_x, y=test_y)
    test_x.tofile(ARTIFACTS / "test_features.f32")

    fp32 = ARTIFACTS / "ds_cnn.onnx"
    if "onnx" in args.targets:
        export_onnx(model, fp32)
        export_onnx_dynamo(model, ARTIFACTS / "ds_cnn_dynamo.onnx")
        export_onnx_fp16(fp32, ARTIFACTS / "ds_cnn_fp16.onnx")
        export_onnx_int8(fp32, ARTIFACTS / "ds_cnn_int8.onnx", calib)
        export_onnx_int8(fp32, ARTIFACTS / "ds_cnn_int8_trt.onnx", calib, for_tensorrt=True)
    if "tflite" in args.targets:
        (ARTIFACTS / "tflite").mkdir(exist_ok=True)
        export_tflite(fp32, ARTIFACTS / "tflite", calib)
    if "tensorrt" in args.targets:
        (ARTIFACTS / "trt").mkdir(exist_ok=True)
        sources = {"fp32": fp32, "fp16": ARTIFACTS / "ds_cnn_fp16.onnx", "int8": ARTIFACTS / "ds_cnn_int8_trt.onnx"}
        for name, src in sources.items():
            build_engine(src, ARTIFACTS / "trt" / f"ds_cnn_{name}.engine")
    if "c" in args.targets:
        q = quant.quantize_model(model, torch.from_numpy(calib))
        write_c_headers(model, ckpt, q, WEIGHTS)
        write_clip_header(clip, ckpt["words"][clip[1]], WEIGHTS)
    print(f"exported {', '.join(args.targets)} -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
