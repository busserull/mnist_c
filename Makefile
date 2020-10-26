SRC := main.c matrix.c mnist.c network.c random.c
CC := gcc
CFLAGS := -DLITTLE_ENDIAN -DDEBUG -g -fsanitize=address -lm
FAST_FLAGS := -DLITTLE_ENDIAN -lm -O3
BUILD := build

a.out : $(SRC:%.c=$(BUILD)/%.o)
	$(CC) $(CFLAGS) $^ -o $@

.PHONY : fast
fast : $(SRC)
	$(CC) $(FAST_FLAGS) $^ -o $@

$(BUILD)/%.o : %.c | $(BUILD)
	$(CC) $(CFLAGS) $< -c -o $@

$(BUILD) :
	mkdir -p $@

.PHONY: clean
clean :
	rm -rf $(BUILD) a.out
