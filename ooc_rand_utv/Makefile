# Definitions of variables.

CC      = gcc
CCFLAGS = -O
LD      = gcc 
LDFLAGS = -O

# Definitions of rules.

simple_test.x : simple_test.o rand_utv_ooc.o NoFLA_UTV_WY_blk_var2.o
	@$(LD)  \
    -o simple_test.x \
    simple_test.o \
	rand_utv_ooc.o \
	NoFLA_UTV_WY_blk_var2.o \
	-Wl,--start-group \
    /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
    /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a \
	/opt/intel/mkl/lib/intel64/libmkl_core.a \
	-Wl,--end-group \
	-ldl -lpthread -lm -lgomp -g

simple_test.o : simple_test.c
	@$(CC) -c simple_test.c \
	-DMKL_LP64 -m64 -g -I/opt/intel/mkl/include

profile.x : profile.o rand_utv_ooc.o
	@$(LD)  \
	-o profile.x \
	profile.o \
	rand_utv_ooc.c \
    -Wl,--start-group \
    /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
    /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a \
    /opt/intel/mkl/lib/intel64/libmkl_core.a \
	-Wl,--end-group \
	-ldl -lpthread -lm -lgomp

profile.o : profile.c
	@$(CC) -c profile.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

block_size_timing.x : block_size_timing.o rand_utv_ooc.o
	@$(LD)  \
	-o block_size_timing.x \
	block_size_timing.o \
	rand_utv_ooc.c \
    -Wl,--start-group \
    /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
    /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a \
    /opt/intel/mkl/lib/intel64/libmkl_core.a \
	-Wl,--end-group \
	-ldl -lpthread -lm -lgomp

block_size_timing.o : block_size_timing.c
	@$(CC) -c block_size_timing.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

timing.x : timing.o rand_utv_ooc.o
	@$(LD)  \
	-o timing.x \
	timing.o \
	rand_utv_ooc.c \
    -Wl,--start-group \
    /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
    /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a \
    /opt/intel/mkl/lib/intel64/libmkl_core.a \
	-Wl,--end-group \
	-ldl -lpthread -lm -lgomp

timing.o : timing.c
	@$(CC) -c timing.c \
	-DMKL_LP64 -m64 -I/opt/intel/mkl/include

%.o : %.c
	$(CC) $(CCFLAGS) -c $< -o $@

clean : FORCE
	rm -f a.out *.x *.o *~ core

FORCE :
