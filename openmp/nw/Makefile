# C compiler
CC = clang++
ICC = clang++
CC_FLAGS = -g -O3 -fopenmp
OFFLOAD_CC_FLAGS = -fopenmp-targets=nvptx64-nvidia-cuda  -Xopenmp-target -march=sm_70 --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}

all: needle needle_offload needle_offload_noreuse

needle: needle.cpp
	$(CC) $(CC_FLAGS) needle.cpp -o needle 

needle_offload: needle.cpp
	$(ICC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD needle.cpp -o needle_offload

needle_offload_noreuse: needle.cpp
	$(ICC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD -DOMP_OFFLOAD_NOREUSE needle.cpp -o needle_offload_noreuse

clean:
	rm -f needle needle_offload
