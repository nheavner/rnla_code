# Definitions of variables

CC      = nvcc
CCFLAGS = -O
LD		= nvcc
LDFLAGS = -O
GENCODE_FLAGS = -arch=sm_35 -gencode arch=compute_35,code=compute_35

# Definitions of rules
simple_test.x : simple_test.o rand_utv_gpu.o 
	@$(LD) $(GENCODE_FLAGS) \
	-o simple_test.x \
	simple_test.o \
	rand_utv_gpu.o \
    -L/usr/local/cuda/lib64 \
	-L/usr/local/magma/lib \
	-lmagma -lcudart -lcusolver -lcublas -lgomp -lcurand \
	-L/opt/intel/mkl/lib/intel64 \
	-lmkl_intel_lp64 -lmkl_core -ldl -lpthread -lgomp -lcublas_device

simple_test.o : simple_test.cu
	@$(CC) $(GENCODE_FLAGS) -c simple_test.cu \
	-I/opt/intel/mkl/include \

rand_utv_gpu.o : rand_utv_gpu.cu
	@$(CC) $(GENCODE_FLAGS) -c rand_utv_gpu.cu \
	-I/opt/intel/mkl/include \
	-I/usr/local/magma/include \
	-dc -DADD_ -DMAGMA_WITH_MKL 

timing_standards.x : timing_standards.o
	@$(LD) $(GENCODE_FLAGS) \
	-o timing_standards.x \
	timing_standards.o \
	-L/usr/local/cuda/lib64 \
	-L/usr/local/magma/lib \
	-lmagma -lcudart -lcusolver -lcublas -lgomp -lcurand \
	-L/opt/intel/mkl/lib/intel64 \
	-lmkl_intel_ilp64 -lmkl_core -ldl -lpthread -lgomp -lcublas_device 

timing_standards.o : timing_standards.cu
	@$(CC) $(GENCODE_FLAGS) -c timing_standards.cu \
	-I/opt/intel/mkl/include \
	-I/usr/local/magma/include \
	-DADD_ -DMKL_ILP64

timing_rutv_gpu.x : timing_rutv_gpu.o rand_utv_gpu.o
	@$(LD) $(GENCODE_FLAGS) \
	-o timing_rutv_gpu.x \
	timing_rutv_gpu.o \
	rand_utv_gpu.o \
	-L/usr/local/cuda/lib64 \
	-L/usr/local/magma/lib \
	-lmagma -lcudart -lcusolver -lcublas -lgomp -lcurand \
	-L/opt/intel/mkl/lib/intel64 \
	-lmkl_intel_lp64 -lmkl_core -ldl -lpthread -lgomp -lcublas_device 

timing_rutv_gpu.o : timing_rutv_gpu.cu
	@$(CC) $(GENCODE_FLAGS) -c timing_rutv_gpu.cu \
	-I/opt/intel/mkl/include \

profile_rutv_gpu.x : profile_rutv_gpu.o rand_utv_gpu.o
	@$(LD) $(GENCODE_FLAGS) \
	-o profile_rutv_gpu.x \
	profile_rutv_gpu.o \
	rand_utv_gpu.o \
	-L/usr/local/cuda/lib64 \
	-L/usr/local/magma/lib \
	-lmagma -lcudart -lcusolver -lcublas -lgomp -lcurand \
	-L/opt/intel/mkl/lib/intel64 \
	-lmkl_intel_lp64 -lmkl_core -ldl -lpthread -lgomp -lcublas_device 

profile_rutv_gpu.o : profile_rutv_gpu.cu
	@$(CC) $(GENCODE_FLAGS) -c profile_rutv_gpu.cu \
	-I/opt/intel/mkl/include \

block_size_timing.x : block_size_timing.o rand_utv_gpu.o
	@$(LD) $(GENCODE_FLAGS) \
	-o block_size_timing.x \
	block_size_timing.o \
	rand_utv_gpu.o \
	-L/usr/local/cuda/lib64 \
	-L/usr/local/magma/lib \
	-lmagma -lcudart -lcusolver -lcublas -lgomp -lcurand \
	-L/opt/intel/mkl/lib/intel64 \
	-lmkl_intel_lp64 -lmkl_core -ldl -lpthread -lgomp -lcublas_device 

block_size_timing.o : block_size_timing.cu
	@$(CC) $(GENCODE_FLAGS) -c block_size_timing.cu \
	-I/opt/intel/mkl/include \

clean : FORCE
	rm -f a.out *.o *~ core

FORCE :
