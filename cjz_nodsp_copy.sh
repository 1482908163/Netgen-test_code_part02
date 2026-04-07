#!/bin/bash
#SBATCH --job-name=Mesh_part
#SBATCH --output=/vol8/home/hnu_lhz/cjz/NETGEN/test_code_part01/err/Mesh_part_r1_mem%j.out
#SBATCH --error=/vol8/home/hnu_lhz/cjz/NETGEN/test_code_part01/err/Mesh_part_r1_mem%j.err
#SBATCH -p mt_module
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1

set -euo pipefail

GCCHOME=/vol8/home/hnu_lhz/cjz/gcc-12
PROJ_DIR=/vol8/home/hnu_lhz/cjz/NETGEN/test_code_part01
BIN_PATH=$PROJ_DIR/build/mesh_occ_mpi/mesh_occ_mpi
INPUT_PATH=$PROJ_DIR/inputData/wholewall3solid.STEP

OUTPUT_PATH=$PROJ_DIR/result/part_r1_mem/
ERR_DIR=$PROJ_DIR/err
LOCAL_LIB=/vol8/home/hnu_lhz/cjz/lib/usr/lib/aarch64-linux-gnu

numlevels=1
numrefine=1
maxh=1000.0
minh=0.0

mkdir -p "$OUTPUT_PATH" 

module purge
module load mpich/mpi-x

# 只清理最容易干扰 MPI/运行库的变量
unset LD_PRELOAD
unset OMPI_MCA_pml
unset OMPI_MCA_pml_ucx_tls
unset PMIX_INSTALL_PREFIX
unset PMIX_MCA_mca_base_component_path

export PATH="$GCCHOME/bin:$PATH"
export LD_LIBRARY_PATH="$GCCHOME/lib64:/vol8/home/hnu_lhz/cjz/NETGEN/install/lib:$LOCAL_LIB:/usr/lib/aarch64-linux-gnu:/vol8/appsoftware/mpi-x/lib"
export OMP_NUM_THREADS=1

echo "=== 开始运行 ==="
echo "开始时间: $(date)"
start_time=$(date +%s)

yhrun --mpi=pmix "$BIN_PATH" \
  -i "$INPUT_PATH" \
  -o "$OUTPUT_PATH" \
  -l "$numlevels" \
  -r "$numrefine" \
  --maxh "$maxh" \
  --minh "$minh" \
  --stream \
  --stream-batch 4096 \
  --stream-vol-batch 4096 \
  --keep-stream-files \
  -v -adj

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "结束时间: $(date)"
echo "运行时间: $runtime 秒"
echo "=== 运行结束 ==="