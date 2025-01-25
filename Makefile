CC = gcc
CFLAGS = -Wall -Iinclude
LDFLAGS = -lblas
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
EXAMPLES = examples/linear_example.c

all: $(OBJ) $(EXAMPLES)
	$(CC) $(OBJ) $(EXAMPLES) -o linear_example $(CFLAGS) $(LDFLAGS)

src/%.o: src/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f src/*.o linear_example