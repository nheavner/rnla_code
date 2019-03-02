
# Definitions of variables.

CC      = gcc
CCFLAGS = -O 
LD      = gcc
LDFLAGS = -O 

# Definitions of rules.

timing.x : timing.o hqrrp_ooc_physical_pivot.o hqrrp_ooc_multithreaded.o 
	@$(LD) $(LDFLAGS) \
            -o timing.x \
            timing.o \
            hqrrp_ooc_physical_pivot.o \
			hqrrp_ooc_multithreaded.o -lm  \
            -Wl,--start-group \
			/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
			/opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a \
			/opt/intel/mkl/lib/intel64/libmkl_core.a \
			-Wl,--end-group \
			-ldl -lpthread -lm -lgomp -fopenmp

timing.o : timing.c
	@$(CC) $(CCFLAGS) -c timing.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

hqrrp_ooc_multithreaded.o : hqrrp_ooc_multithreaded.c
	@$(CC) $(CCFLAGS) -c hqrrp_ooc_multithreaded.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

hqrrp_ooc_physical_pivot.o : hqrrp_ooc_physical_pivot.c
	@$(CC) $(CCFLAGS) -c hqrrp_ooc_physical_pivot.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

clean: 
	rm -f a.out *.x *.o *~ core

