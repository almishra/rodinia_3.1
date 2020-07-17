# C compiler
CC = g++
ICC = g++
CLANG = clang++
CC_FLAGS = -g -O3 -fopenmp
OFFLOAD_CC_FLAGS = -foffload=nvptx-none
CLANG_FLAGS = -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70

all: needle needle_offload needle_offload_clang

needle: 
	$(CC) $(CC_FLAGS) needle.cpp -o needle 

needle_offload:
	$(ICC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD needle.cpp -o needle_offload

needle_offload_clang:
	$(CLANG) $(CC_FLAGS) $(CLANG_FLAGS) -DOMP_OFFLOAD needle.cpp -o needle_offload_clang

clean:
	rm -f needle needle_offload
