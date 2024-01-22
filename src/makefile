.SECONDEXPANSION:

MLP_NEUROSIM = mlp_neurosim

BUILD_DIR = ../build

ALLSRC := $(wildcard $(MLP_NEUROSIM)/*.cpp $(MLP_NEUROSIM)/NeuroSim/*.cpp)

MAINS := main.cpp
SRC := $(filter-out $(MAINS),$(ALLSRC))
OBJ := $(SRC:.cpp=)
ALLOBJ := $(patsubst %, $(BUILD_DIR)/%.o, $(OBJ))

$(info    Creating following objects $(ALLOBJ))

CXX := g++
CXXFLAGS := -fopenmp -O3 -std=c++0x -w -fPIC

.PHONY: all clean

all: libmlp.so
libmlp.so: $(ALLOBJ)
	$(CXX) $(CXXFLAGS) $^ -shared -o $(BUILD_DIR)/$@

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $@/mlp_neurosim/NeuroSim

clean:
	$(RM) mlp.so
	$(RM) $(ALLOBJ)
