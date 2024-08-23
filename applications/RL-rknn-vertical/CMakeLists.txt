cmake_minimum_required(VERSION 3.10)

project(rknn_test)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_test ./src/main.cpp)

set(sources ${CMAKE_SOURCE_DIR}/src/main.cpp ${CMAKE_SOURCE_DIR}/src/ADS1x15.cpp) 
target_sources(rknn_test PUBLIC ${sources})


set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/lib/librknnrt.so)

target_link_libraries(rknn_test ${RKNN_RT_LIB})

target_link_libraries(rknn_test ${CMAKE_SOURCE_DIR}/imu/MotionSensor/libMotionSensor.a  ${CMAKE_SOURCE_DIR}/imu/libs/libI2Cdev.a)

target_link_libraries(rknn_test -lpthread -lwiringPi -lcrypt -lm -lrt -fconcepts -li2c)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_test_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_test DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
