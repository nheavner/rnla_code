
# Definitions of variables.

CC      = gcc
CCFLAGS = -O 
LD      = gcc
LDFLAGS = -O 

# Definitions of rules.

simple_test.x : simple_test.o hqrrp_ooc.o NoFLA_HQRRP_WY_blk_var4.o
	@$(LD) $(LDFLAGS) \
            -o simple_test.x \
            simple_test.o \
            hqrrp_ooc.c -lm  \
			NoFLA_HQRRP_WY_blk_var4.c \
            -Wl,--start-group \
			/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
			/opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a \
			/opt/intel/mkl/lib/intel64/libmkl_core.a \
			-Wl,--end-group \
			-ldl -lpthread -lm -lgomp -fopenmp

timing.x : timing.o hqrrp_ooc.o NoFLA_HQRRP_WY_blk_var4.o
	@$(LD) $(LDFLAGS) \
            -o timing.x \
            timing.o \
            hqrrp_ooc.o \
			NoFLA_HQRRP_WY_blk_var4.c \
            -Wl,--start-group \
			/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
			/opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a \
			/opt/intel/mkl/lib/intel64/libmkl_core.a \
			-Wl,--end-group \
			-ldl -lpthread -lm -lgomp -fopenmp

block-size-test.x : block-size-test.o hqrrp_ooc.o NoFLA_HQRRP_WY_blk_var4.o
	@$(LD) $(LDFLAGS) \
            -o block-size-test.x \
            block-size-test.o \
            hqrrp_ooc.c -lm  \
			NoFLA_HQRRP_WY_blk_var4.c \
            -Wl,--start-group \
			/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
			/opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a \
			/opt/intel/mkl/lib/intel64/libmkl_core.a \
			-Wl,--end-group \
			-ldl -lpthread -lm -lgomp -fopenmp

simple_test.o : simple_test.c
	@$(CC) $(CCFLAGS) -c simple_test.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

timing.o : timing.c
	@$(CC) $(CCFLAGS) -c timing.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

block-size-test.o : block-size-test.c
	@$(CC) $(CCFLAGS) -c block-size-test.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

hqrrp_ooc.o : hqrrp_ooc.c
	@$(CC) $(CCFLAGS) -c hqrrp_ooc.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

NoFLA_HQRRP_WY_blk_var4.o : NoFLA_HQRRP_WY_blk_var4.c
	@$(CC) $(CCFLAGS) -c NoFLA_HQRRP_WY_blk_var4.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

clean: 
	rm -f a.out *.x *.o *~ core

