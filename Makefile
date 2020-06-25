SRC := main.c matrix.c
CC := gcc
CFLAGS := -DDEBUG -g -fsanitize=address
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
