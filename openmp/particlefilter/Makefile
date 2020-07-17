#makefile
all: openmp offload_gcc offload_clang

openmp: ex_particle_OPENMP_seq.cpp
	g++ -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.cpp -o particle_filter -lm 

offload_gcc: ex_particle_OPENMP_seq.cpp
	g++ -O3 -ffast-math -DOMP_OFFLOAD -fopenmp  -foffload=nvptx-none ex_particle_OPENMP_seq.cpp -o particle_filter_offload_gcc -lm 

offload_clang: ex_particle_OPENMP_seq.cpp
	clang++ -O3 -ffast-math -DOMP_OFFLOAD -fopenmp   -fopenmp-targets=nvptx64-nvidia-cuda  -Xopenmp-target -march=sm_70 ex_particle_OPENMP_seq.cpp -o particle_filter_offload_clang -lm 


#icc_openmp: ex_particle_OPENMP_seq.c
#	icc -O3 -fast -openmp ex_particle_OPENMP_seq.c -o particle_filter



clean:
	rm particle_filter
