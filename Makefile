CC = gcc
CFLAGS = -Wall -Iinclude
LDFLAGS = -lblas -lm

SRC = $(wildcard src/*/*.c)
OBJ = $(SRC:.c=.o)

EXAMPLES = $(wildcard examples/*.c)
EXECS = $(EXAMPLES:.c=.out)  # Converts .c to executable names without extension + .out

all: $(OBJ) $(EXECS)

examples/%: examples/%.c $(OBJ)
	$(CC) $(OBJ) $< -o $@ $(CFLAGS) $(LDFLAGS)

src/%/%.o: src/%/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f examples/linear_example examples/linear_relu_example
	find . -name "*.o" -type f -delete