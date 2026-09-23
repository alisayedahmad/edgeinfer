# the whole pipeline, phase by phase. `make help` lists the targets.
PYTHON ?= python3
BENCHMARK_URL := https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model

.PHONY: help data train export smoke-export c-engine test bench bench-onnx bench-tflite bench-trt bench-c \
        analysis arm qemu tools clean bench-torch

help:
	@grep -hE '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:[^#]*## /\t/' | expand -t 16

data: ## download speech commands v2 and build the audio caches
	$(PYTHON) -m data.speech_commands

train: ## train ds-cnn on the caches
	$(PYTHON) -m train.train

export: ## export onnx, tflite, tensorrt engines and the c headers
	$(PYTHON) -m train.export

smoke-export: ## export an untrained model on synthetic features, no dataset needed
	$(PYTHON) -m train.export --random-model --synthetic --targets onnx c --calib 256

c-engine: ## build the c engine cli and run its operator tests
	$(MAKE) -C c_engine
	$(MAKE) -C c_engine test

test: ## python tests
	$(PYTHON) -m pytest

bench-torch: ## the pytorch reference
	$(PYTHON) -m runtime.torch_runner

bench-onnx: ## onnx runtime, fused and unfused, fp32 and int8
	$(PYTHON) -m runtime.onnx_runner --precision fp32
	$(PYTHON) -m runtime.onnx_runner --precision fp32-unopt
	$(PYTHON) -m runtime.onnx_runner --precision int8

bench-tflite: ## tflite fp32 and int8
	$(PYTHON) -m runtime.tflite_runner --precision fp32
	$(PYTHON) -m runtime.tflite_runner --precision int8

bench-trt: ## tensorrt fp32, fp16 and int8, needs a supported gpu
	$(PYTHON) -m runtime.tensorrt_runner --precision fp32
	$(PYTHON) -m runtime.tensorrt_runner --precision fp16
	$(PYTHON) -m runtime.tensorrt_runner --precision int8

bench-c: ## the c engine, fused, unfused and int8
	$(PYTHON) -m runtime.c_runner --precision fp32
	$(PYTHON) -m runtime.c_runner --precision fp32-unfused
	$(PYTHON) -m runtime.c_runner --precision int8

# the optional ones keep going if a runtime is not installed on this machine
bench: bench-torch bench-onnx bench-c ## every runtime this machine can run
	-$(MAKE) bench-tflite
	-$(MAKE) bench-trt

analysis: ## tables and plots from whatever results exist
	$(PYTHON) -m analysis.compare_runtimes
	$(PYTHON) -m analysis.fusion_analysis
	$(PYTHON) -m analysis.memory_timeline --all-modes
	$(PYTHON) -m analysis.quant_error
	$(PYTHON) -m analysis.mfcc_error

arm: ## cross compile for cortex-m4
	$(MAKE) -C c_engine arm

qemu: arm ## run the cortex-m4 build under qemu, check flash and ram
	$(PYTHON) embedded/constraints.py --qemu

tools: tools/benchmark_model ## fetch tflite's per-op profiler

tools/benchmark_model:
	mkdir -p tools
	curl -fsSL -o $@ $(BENCHMARK_URL)
	chmod +x $@

clean: ## drop build output, keep the dataset and artifacts
	$(MAKE) -C c_engine clean
	rm -rf .pytest_cache **/__pycache__
