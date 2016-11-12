# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 
LD_FLAGS = -lm

histo: 
	$(CC) $(CC_FLAGS) main.c utils.c dump.c -o histo $(LD_FLAGS) 

clean:
	rm -f histo *.bmp

.PHONY: histo
