NVCC = nvcc
NVCC_FLAGS = --gpu-architecture=sm_80 -m64
SRC := src

all: transpose.o transpose

transpose.o: $(SRC)/transpose.cu
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/transpose.cu -o transpose.o

transpose: transpose.o
	$(NVCC) $(NVCC_FLAGS) transpose.o -o transpose
