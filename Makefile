CC = gcc
CFLAGS = -Wall -Iinclude -mavx -DENABLE_AVX2 -O3
DEBUG_CFLAGS = -Wall -Iinclude -mavx -DENABLE_AVX2 -g
LDFLAGS = -lblas -lm

SRC = $(shell find src -name "*.c" -type f)
OBJ = $(SRC:.c=.o)
DEBUG_OBJ = $(SRC:.c=.debug.o)

EXAMPLES = $(wildcard examples/*.c)
EXECS = $(EXAMPLES:.c=.out)  # Converts .c to executable names without extension + .out
DEBUG_EXECS = $(EXAMPLES:.c=.debug.out)

all: $(OBJ) $(EXECS)

examples/%.out: examples/%.c $(OBJ)
	$(CC) $(OBJ) $< -o $@ $(CFLAGS) $(LDFLAGS)

src/%.o: src/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

debug: $(DEBUG_OBJ) $(DEBUG_EXECS)

examples/%.debug.out: examples/%.c $(DEBUG_OBJ)
	$(CC) $(DEBUG_OBJ) $< -o $@ $(DEBUG_CFLAGS) $(LDFLAGS)

src/%.debug.o: src/%.c
	$(CC) -c $< -o $@ $(DEBUG_CFLAGS)

clean:
	find . -name "*.o" -type f -delete
	find . -name "*.out" -type f -delete