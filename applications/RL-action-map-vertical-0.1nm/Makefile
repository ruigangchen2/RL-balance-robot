CXX=g++
STANDARD=-std=c++17 
WNO_FLAG=-Wno-psabi

PROG=run

%.o: %.c                                                                         
	$(CXX) $(STANDARD) $(WNO_FLAG) $< -o $@


all: $(PROG).o
	$(CXX) $(STANDARD) $(WNO_FLAG) -o $(PROG) \
		src/main.cpp \
		src/ADS1220.c \
		imu/MotionSensor/libMotionSensor.a \
		imu/libs/libI2Cdev.a \
		-lboost_system \
		-I /usr/include/hdf5/serial \
		-L /usr/lib/aarch64-linux-gnu/hdf5/serial \
		-lhdf5 \
		-lhdf5_cpp \
		-lpthread \
		-lwiringPi \
		-lcrypt \
		-lm \
		-lrt \
		-fconcepts \
		-li2c

$(PROG).o: IMUlibrary.a I2Clibrary.a 

IMUlibrary.a:
	$(MAKE) -C imu/MotionSensor/ 

I2Clibrary.a:
	$(MAKE) -C imu/libs/I2Cdev

clean:
	cd imu/MotionSensor && $(MAKE) clean
	cd imu/libs/I2Cdev && $(MAKE) clean
	rm -rf *.o *~ *.mod
	rm -rf $(PROG)
