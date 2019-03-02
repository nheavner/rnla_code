# Definitions of variables

CC      = nvcc
CCFLAGS = -O
LD		= nvcc
LDFLAGS = -O
GENCODE_FLAGS = -arch=sm_35 -gencode arch=compute_35,code=compute_35
DEBUG_FLAGS = -g

# Definitions of rules
magma_qr_timing.x : magma_qr_timing.o 
	@$(LD) $(GENCODE_FLAGS) $(DEBUG_FLAGS) \
	-o magma_qr_timing.x \
	magma_qr_timing.o \
	-L/usr/local/magma/lib \
	-lmagma -lcusolver -lcublas -lgomp \
	-L/opt/intel/mkl/lib/intel64 \
	-lmkl_intel_ilp64 -lmkl_core -ldl -lpthread -lgomp

magma_qr_timing.o : magma_qr_timing.cu
	@$(CC) $(GENCODE_FLAGS) $(DEBUG_FLAGS) -c magma_qr_timing.cu \
	-I/opt/intel/mkl/include -I/usr/local/magma/include -dc -DADD_ \
	-DMAGMA_WITH_MKL

clean : FORCE
	rm -f a.out *.o *~ core

FORCE :
