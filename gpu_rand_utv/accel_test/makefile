# Definitions of variables

CC      = nvcc
CCFLAGS = -O
LD		= nvcc
LDFLAGS = -O
GENCODE_FLAGS = -arch=sm_35 -gencode arch=compute_35,code=compute_35

# Definitions of rules
timing_gpu.x : timing_gpu.o rand_utv_gpu.o
	@$(LD) $(GENCODE_FLAGS) \
	-o timing_gpu.x \
	timing_gpu.o \
	rand_utv_gpu.o \
	-L/usr/local/cuda/lib64 \
	-L/usr/local/magma/lib \
	-lmagma -lcudart -lcusolver -lcublas -lgomp -lcurand \
	-L/opt/intel/mkl/lib/intel64 \
	-lmkl_intel_lp64 -lmkl_core -ldl -lpthread -lgomp -lcublas_device 

timing_cpu.x : timing_cpu.o NoFLA_UTV_WY_blk_var2.o
	@gcc -o timing_cpu.x \
	timing_cpu.o NoFLA_UTV_WY_blk_var2.o \
	-Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
	/opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a \
	/opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group \
	-ldl -lpthread -lm -lgomp

timing_gpu.o : timing_gpu.cu
	@$(CC) $(GENCODE_FLAGS) -c timing_gpu.cu \
	-I/opt/intel/mkl/include \
	-I/usr/local/magma/include \
	-dc -DADD_ -DMAGMA_WITH_MKL

rand_utv_gpu.o : ../rand_utv_gpu.cu
	@$(CC) $(GENCODE_FLAGS) -c ../rand_utv_gpu.cu \
	-I/opt/intel/mkl/include \
	-I/usr/local/magma/include \
	-dc -DADD_ -DMAGMA_WITH_MKL 

timing_cpu.o : timing_cpu.c 
	@gcc -c timing_cpu.c

NoFLA_UTV_WY_blk_var2.o : ../../rand_utv/NoFLA_UTV_WY_blk_var2.c
	@gcc -c ../../rand_utv/NoFLA_UTV_WY_blk_var2.c \
	-I/opt/intel/mkl/include

clean : FORCE
	rm -f a.out *.o *~ core

FORCE :
