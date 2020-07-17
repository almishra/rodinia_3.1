LOCAL_CC = gcc -g -O3 -Wall
CC = gcc
CFLAGS = -lm -fopenmp -Wall

all : nn nn_offload_gcc nn_offload_clang

clean :
	rm -rf *.o nn nn_offload_*

nn : nn_openmp.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) 

nn_offload_gcc : nn_openmp.c
	$(CC) -foffload=nvptx-none -DOMP_OFFLOAD -o $@ $< $(LDFLAGS) $(CFLAGS) 

nn_offload_clang : nn_openmp.c
	clang  -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70 -DOMP_OFFLOAD -o $@ $< $(LDFLAGS) $(CFLAGS) 

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

