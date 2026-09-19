# native cli + tests, and the cortex-m4 build for qemu
#
#   make          build/host/edgeinfer, needs weights/ from `python -m train.export --targets c`
#   make test     operator tests, vectors from test/gen_vectors.py
#   make arm      build/arm/edgeinfer.elf, int8 + fused fp32, no fpu

PYTHON ?= python3
CC ?= cc
OPT ?= -O3 -march=native
CFLAGS += $(OPT) -std=c11 -Wall -Wextra -Iinclude -Iweights
BUILD := build

LIB_SRC := src/memory.c src/quantize.c src/mfcc.c $(wildcard src/ops/*.c)
SRC := $(LIB_SRC) src/forward.c
HEADERS := $(wildcard include/*.h)
WEIGHTS := weights/model.h weights/geometry.h weights/weights_f32.h weights/weights_raw.h weights/weights_int8.h

ARM_CC := arm-none-eabi-gcc
ARM_CFLAGS := -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -O2 -std=c11 -Wall -Wextra \
	-ffunction-sections -fdata-sections -Iinclude -Iweights -I$(BUILD)/arm -DEI_WITH_UNFUSED=0
ARM_LDFLAGS := -nostartfiles -T link.ld -Wl,--gc-sections -Wl,-Map=$(BUILD)/arm/edgeinfer.map --specs=nano.specs

.PHONY: all test arm clean

all: $(BUILD)/host/edgeinfer

$(BUILD)/host/edgeinfer: $(SRC) src/main.c $(HEADERS) $(WEIGHTS) weights/mfcc_tables.h | $(BUILD)/host
	$(CC) $(CFLAGS) -o $@ $(SRC) src/main.c -lm

$(BUILD)/host/test_ops: test/test_ops.c $(LIB_SRC) $(HEADERS) test/vectors.h weights/mfcc_tables.h | $(BUILD)/host
	$(CC) $(CFLAGS) -Itest -o $@ test/test_ops.c $(LIB_SRC) -lm

test: $(BUILD)/host/test_ops
	./$<

weights/mfcc_tables.h:
	cd .. && $(PYTHON) -m train.export --targets mfcc

test/vectors.h: test/gen_vectors.py ../train/quant.py
	cd .. && $(PYTHON) -m c_engine.test.gen_vectors

$(WEIGHTS):
	@echo "missing $@: run python -m train.export --targets c from the repo root" && false

# the arena is sized from the host planner, so ram use is known at link time
$(BUILD)/arm/plan.h: $(BUILD)/host/edgeinfer | $(BUILD)/arm
	./$< header > $@

$(BUILD)/arm/edgeinfer.elf: $(SRC) $(HEADERS) $(WEIGHTS) ../embedded/main.c ../embedded/startup.s link.ld $(BUILD)/arm/plan.h
	$(ARM_CC) $(ARM_CFLAGS) $(ARM_LDFLAGS) -o $@ ../embedded/startup.s ../embedded/main.c $(SRC)

arm: $(BUILD)/arm/edgeinfer.elf
	arm-none-eabi-size $<

$(BUILD)/host $(BUILD)/arm:
	mkdir -p $@

clean:
	rm -rf $(BUILD)
