"""export checks: the onnx graph must match pytorch and keep its layer names."""
import numpy as np
import pytest
import torch

from train import export
from train.ds_cnn import DSCNN, N_FRAMES, N_MFCC

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = DSCNN(n_classes=12, width=16, blocks=2).eval()
    # non-trivial bn statistics, otherwise folding is the identity
    with torch.no_grad():
        for module in m.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.normal_(0, 0.5)
                module.running_var.uniform_(0.5, 2.0)
                module.weight.uniform_(0.5, 1.5)
                module.bias.normal_(0, 0.5)
    return m


@pytest.fixture(scope="module")
def exported(model, tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "ds_cnn.onnx"
    export.export_onnx(model, path)
    return path


def test_onnx_matches_torch(model, exported):
    x = torch.randn(8, 1, N_FRAMES, N_MFCC)
    sess = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"mfcc": x.numpy()})[0]
    assert np.abs(got - model(x).detach().numpy()).max() < 1e-4


def test_graph_keeps_layers_unfused(exported):
    nodes = onnx.load(exported).graph.node
    names = [n.name for n in nodes]
    assert names[:3] == ["conv1", "conv1_bn", "conv1_relu"]
    assert names[-3:] == ["pool", "flatten", "fc"]
    # bn must survive, otherwise the runtimes cannot show their own fusion
    assert sum(n.op_type == "BatchNormalization" for n in nodes) == 5


def test_dynamic_batch(exported):
    sess = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    for batch in (1, 3, 16):
        out = sess.run(None, {"mfcc": np.zeros((batch, 1, N_FRAMES, N_MFCC), np.float32)})[0]
        assert out.shape == (batch, 12)


def test_fp16_copy_runs(exported, tmp_path):
    out = tmp_path / "fp16.onnx"
    export.export_onnx_fp16(exported, out)
    model = onnx.load(out)
    assert {n.op_type for n in model.graph.node} >= {"Cast", "Conv"}
    x = np.random.randn(4, 1, N_FRAMES, N_MFCC).astype(np.float32)
    sess = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    fp16 = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {"mfcc": x})[0]
    got = fp16.run(None, {"mfcc": x})[0]
    assert np.abs(got - ref).max() < 0.05 * (ref.max() - ref.min())


def test_int8_qdq_is_symmetric_for_tensorrt(exported, tmp_path):
    from onnx import numpy_helper

    calib = np.random.randn(64, 1, N_FRAMES, N_MFCC).astype(np.float32)
    out = tmp_path / "int8_trt.onnx"
    export.export_onnx_int8(exported, out, calib, for_tensorrt=True)
    graph = onnx.load(out).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    zeros = [inits[n.input[2]] for n in graph.node if n.op_type == "QuantizeLinear" and len(n.input) > 2]
    assert zeros and all((z == 0).all() for z in zeros)
    # tensorrt's parser wants float bias, not an int32 dequantize
    biases = [inits[n.input[2]].dtype for n in graph.node
              if n.op_type == "Conv" and len(n.input) > 2 and n.input[2] in inits]
    assert all(dtype == np.float32 for dtype in biases)


def test_c_headers_written(model, tmp_path):
    from train import quant

    calib = torch.randn(32, 1, N_FRAMES, N_MFCC)
    ckpt = {"feat_mean": np.zeros(N_MFCC, np.float32), "feat_std": np.ones(N_MFCC, np.float32)}
    export.write_mfcc_tables(tmp_path)
    export.write_c_headers(model, ckpt, quant.quantize_model(model, calib), tmp_path)
    export.write_clip_header((np.zeros(16000, np.int16), 3), "cat", tmp_path)
    names = {p.name for p in tmp_path.glob("*.h")}
    assert names == {"mfcc_tables.h", "model.h", "geometry.h", "weights_f32.h", "weights_raw.h",
                     "weights_int8.h", "clip.h"}
    model_h = (tmp_path / "model.h").read_text()
    assert "#define EI_WIDTH 16" in model_h and "#define EI_LAYERS 5" in model_h
