
# Definitions of variables.

CC      = gcc
CCFLAGS = -O 
LD      = gcc
LDFLAGS = -O 

# Definitions of rules.

hard_drive_test.x : hard_drive_test.o 
	@$(LD) $(LDFLAGS) \
            -o hard_drive_test.x \
            hard_drive_test.o \

hard_drive_test.o : hard_drive_test.c
	@$(CC) $(CCFLAGS) -c hard_drive_test.c \

clean: 
	rm -f a.out *.x *.o *~ core

