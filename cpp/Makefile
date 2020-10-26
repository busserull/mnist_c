SRC := $(shell find *.cpp)
CC := g++
CFLAGS := -g -fsanitize=address
BUILD := build

a.out : $(SRC:%.cpp=$(BUILD)/%.o)
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD) :
	mkdir -p $@

$(BUILD)/%.o : %.cpp | $(BUILD)
	$(CC) $(CFLAGS) $< -c -o $@

.PHONY: clean
clean :
	rm -rf $(BUILD) a.out
