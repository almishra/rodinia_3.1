# C compiler
CC = clang++
ICC = clang++
CC_FLAGS = -g -fopenmp -O2
GPU_FLAGS = -fopenmp-targets=nvptx64-nvidia-cuda  -Xopenmp-target -march=sm_70 --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}

all: bfs bfs_offload bfs_offload_noreuse

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs 

bfs_offload: bfs.cpp
	$(ICC) $(CC_FLAGS) $(GPU_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

bfs_offload_noreuse: bfs.cpp
	$(ICC) $(CC_FLAGS) $(GPU_FLAGS) -DOMP_OFFLOAD -DOMP_OFFLOAD_NOREUSE bfs.cpp -o bfs_offload_noreuse

clean:
	rm -f bfs bfs_offload result.txt
