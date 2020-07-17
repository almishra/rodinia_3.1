# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -fopenmp -foffload=nvptx-none
CLANG = clang++
CLANG_FLAGS = -g -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70

all: bfs bfs_offload_gcc bfs_offload_clang

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs 

bfs_offload_gcc: bfs.cpp
	$(CC) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload_gcc

bfs_offload_clang: bfs.cpp
	$(CLANG) $(CLANG_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload_clang

clean:
	rm -f bfs bfs_offload_gcc bfs_offload_clang result.txt
