#Author: Paritosh Ramanan
#Date: January 21st 2024

.SECONDEXPANSION:

MLP_NEUROSIM = src/mlp_neurosim
BUILD_DIR = ./build
ALLSRC := $(wildcard $(MLP_NEUROSIM)/*.cpp $(MLP_NEUROSIM)/NeuroSim/*.cpp)

MAINS := main.cpp
SRC := $(filter-out $(MAINS),$(ALLSRC))
OBJ := $(SRC:.cpp=)
ALLOBJ := $(patsubst src/%, $(BUILD_DIR)/%.o, $(OBJ))

$(info    Creating following objects $(ALLOBJ))

CXX := -static-libstdc++ g++
CXXFLAGS := -fopenmp -O3 -std=c++0x -w -fPIC

.PHONY: all clean

all: create-build neurosim meliso

neurosim: $(ALLOBJ)
	$(CXX) $(CXXFLAGS) $^ -shared -o $(BUILD_DIR)/libmlp.so

$(BUILD_DIR)/%.o: src/%.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

create-build:
	mkdir -p $(BUILD_DIR)/mlp_neurosim/NeuroSim

meliso:
	python3 setup.py build_ext --inplace --build-lib $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

clean-neurosim:
	$(RM) $(BUILD_DIR)/libmlp.so $(ALLOBJ)
	rm -rf $(BUILD_DIR)/src

clean-meliso:
	python3 setup.py clean --all
	$(RM) $(BUILD_DIR)/meliso.*