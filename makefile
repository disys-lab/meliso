#Author: Paritosh Ramanan
#Date: January 21st 2024

.SECONDEXPANSION:
IMMV_REPO = https://github.com/disys-lab/InMemMVM
IMMV_DIR  = .inmemmvm_tmp
MLP_NEUROSIM = src/mlp_neurosim
BUILD_DIR = ./build
ALLSRC := $(wildcard $(MLP_NEUROSIM)/*.cpp $(MLP_NEUROSIM)/NeuroSim/*.cpp)

MAINS := main.cpp
SRC := $(filter-out $(MAINS),$(ALLSRC))
OBJ := $(SRC:.cpp=)
ALLOBJ := $(patsubst src/%, $(BUILD_DIR)/%.o, $(OBJ))

$(info    Creating following objects $(ALLOBJ))

CXX := g++ -static-libstdc++ 
CXXFLAGS := -fopenmp -O3 -std=c++0x -w -fPIC

.PHONY: all clean update-immv clean-immv

all: get-immv create-build neurosim meliso

get-immv:
	@echo "Cloning or updating InMemMVM..."
	@if [ -d "$(IMMV_DIR)/.git" ]; then \
		echo "Repo exists — pulling latest changes..."; \
		cd $(IMMV_DIR) && git pull; \
	else \
		rm -rf $(IMMV_DIR); \
		git clone $(IMMV_REPO) $(IMMV_DIR); \
	fi
	@echo "Copying files into $(MLP_NEUROSIM)..."
	@mkdir -p $(MLP_NEUROSIM)
	@cp -a $(IMMV_DIR)/. $(MLP_NEUROSIM)/
	@rm -rf $(MLP_NEUROSIM)/.git
	@echo "Done."

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

clean-immv:
	@echo "Removing temp repo folder..."
	rm -rf $(IMMV_DIR)
	rm -rf $(MLP_NEUROSIM)