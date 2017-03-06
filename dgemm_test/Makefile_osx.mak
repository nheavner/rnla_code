# Definitions of variables

CC = icc# gcc-6
CCFLAGS = -O
LD = icc# gcc-6
LDFLAGS = -O

# Definitions of rules

dgemm_test.x : dgemm_test.o
	$(LD)  \
	-o dgemm_test.x \
	dgemm_test.o \
	/opt/intel/mkl/lib/libmkl_intel_ilp64.a \
	/opt/intel/mkl/lib/libmkl_intel_thread.a \
	/opt/intel/mkl/lib/libmkl_core.a \
	-L/opt/intel/compilers_and_libraries_2017.2.163/mac/compiler/lib \
	-liomp5 -lpthread -lm -ldl
	
	install_name_tool -change @rpath/libiomp5.dylib /opt/intel/compilers_and_libraries_2017.2.163/mac/compiler/lib/libiomp5.dylib ./dgemm_test.x


dgemm_test.o : dgemm_test.c
	$(CC) -c dgemm_test.c \
	-DMKL_ILP64 -m64 -I/opt/intel/mkl/include

clean:
	rm -f a.out *.x *.o *~ core
        
