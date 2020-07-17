# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O3

all: srad srad_offload_gcc srad_offload_clang

srad: srad.cpp
	$(CC) $(CC_FLAGS) srad.cpp -o srad 

srad_offload_gcc: srad.cpp
	$(CC) $(CC_FLAGS) -DOMP_OFFLOAD  -foffload=nvptx-none srad.cpp -o srad_offload_gcc

srad_offload_clang: srad.cpp
	clang++ $(CC_FLAGS) -DOMP_OFFLOAD -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70 srad.cpp -o srad_offload_clang

clean:
	rm -f srad srad_offload_gcc srad_offload_clang
