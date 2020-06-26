SRC := main.c matrix.c mnist.c network.c
CC := gcc
CFLAGS := -DLITTLE_ENDIAN -DDEBUG -g -fsanitize=address -lm
BUILD := build

a.out : $(SRC:%.c=$(BUILD)/%.o)
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD)/%.o : %.c | $(BUILD)
	$(CC) $(CFLAGS) $< -c -o $@

$(BUILD) :
	mkdir -p $@

.PHONY: clean
clean :
	rm -rf $(BUILD) a.out
