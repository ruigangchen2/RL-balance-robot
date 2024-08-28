CXX = g++ 
C=gcc -c
STANDARD=-std=c++17 
WNO_FLAG=-Wno-psabi

#DEBUGGING FLAGS
PTHREAD=-lpthread
CXXFLAGS=-Wall -g3
LIB_CURL=-lcurl
LIB_FS=-lstdc++fs
FCONCEPTS=-fconcepts

#COMMON PARAMS
COMMON_FLAGS=$(CXX) $(STANDARD) $(WNO_FLAG) $(FCONCEPTS)
DEBUG_FLAG=BUILD_RELEASE


.PHONY: RL-output 

all: RL-output 
RL-output: src/main.cpp
	$(COMMON_FLAGS)  src/main.cpp src/ADS1x15.cpp -lboost_system -I /usr/include/hdf5/serial -L /usr/lib/aarch64-linux-gnu/hdf5/serial -lhdf5 -lhdf5_cpp -lpthread -lwiringPi -lcrypt -lm -lrt -fconcepts -li2c -o run.o


