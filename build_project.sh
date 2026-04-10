#!/bin/bash
set -e

PROJ_DIR=/vol8/home/hnu_lhz/cjz/NETGEN/test_code_part02
BUILD_DIR=$PROJ_DIR/build
GCCHOME=/vol8/home/hnu_lhz/cjz/gcc-12
LOCAL_LIB=/vol8/home/hnu_lhz/cjz/lib/usr/lib/aarch64-linux-gnu

echo ">>> 加载必要的模块..."
module purge
module load mpich/mpi-x

echo ">>> 设置 gcc-12 环境..."
export PATH=$GCCHOME/bin:$PATH
export LD_LIBRARY_PATH=$GCCHOME/lib64:$LD_LIBRARY_PATH

# 让 MPICH wrapper 底层使用 gcc-12
export MPICH_CC=$GCCHOME/bin/gcc
export MPICH_CXX=$GCCHOME/bin/g++

# 链接阶段和运行阶段都补库路径
export LIBRARY_PATH=$LOCAL_LIB:/usr/lib/aarch64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=$LOCAL_LIB:/usr/lib/aarch64-linux-gnu:/vol8/home/hnu_lhz/cjz/NETGEN/install/lib:/vol8/home/hnu_lhz/cjz/install_libs/lib:$LD_LIBRARY_PATH

echo ">>> 检查MPI编译器..."
which mpicc
which mpicxx

echo ">>> mpicc -show"
mpicc -show
echo ">>> mpicxx -show"
mpicxx -show

echo ">>> 显示当前环境..."
module list
echo "LIBRARY_PATH: $LIBRARY_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo ">>> 清理旧 build 目录..."
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo ">>> 运行 CMake 配置..."
cmake \
  -DCMAKE_INSTALL_PREFIX=../../install-test \
  -DCMAKE_C_COMPILER=$(which mpicc) \
  -DCMAKE_CXX_COMPILER=$(which mpicxx) \
  .. || { echo "❌ CMake 配置失败"; exit 1; }

echo ">>> 开始编译..."
make -j$(nproc) || { echo "❌ Make 编译失败"; exit 1; }

echo ">>> 开始安装..."
make install || { echo "❌ Make install 失败"; exit 1; }

echo "✅ 编译完成，安装路径在: $PROJ_DIR/install-test"
