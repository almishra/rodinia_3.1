# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -fopenmp -O2
#OFFLOAD_CC_FLAGS = -offload-option,mic,compiler,"-no-opt-prefetch"
OFFLOAD_CC_FLAGS = -foffload=nvptx-none

all: hotspot hotspot_offload_gcc hotspot_offload_clang


hotspot: hotspot_openmp.cpp Makefile 
	$(CC) $(CC_FLAGS) hotspot_openmp.cpp -o hotspot 

hotspot_offload_gcc: hotspot_openmp.cpp Makefile
	$(CC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD hotspot_openmp.cpp -o hotspot_offload_gcc

hotspot_offload_clang: hotspot_openmp.cpp Makefile
	clang++ $(CC_FLAGS) -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70 -DOMP_OFFLOAD hotspot_openmp.cpp -o hotspot_offload_clang

clean:
	rm -f hotspot hotspot_offload
