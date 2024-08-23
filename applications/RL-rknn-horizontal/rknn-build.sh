set ff=UNIX
set -e
TARGET_SOC="rk3588" # set the target platform
export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH 
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64
if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi
cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=${TARGET_SOC}
make -j4
make install
cd -
