#!/bin/bash

#SBATCH --job-name=Mesh_test # 程序的作业名
#SBATCH --output=Mesh_test_job1.out         # 标准输出文件
#SBATCH --error=Mesh_test_job1.err          # 错误输出文件
#SBATCH --time=00:15:00               # 最长运行时间 (HH:MM:SS)
#SBATCH --partition=mt_test         # 使用的分区 (更新为 mt_module)            
#SBATCH --nodes=2                    # 请求的节点数
#SBATCH --ntasks-per-node=2         # 每个节点上的任务数 (进程数) MAX 16
#SBATCH --ntasks=4                  # 请求的任务数 (总核数)
#SBATCH --exclude=cn7833,cn7836  


export filename=wholewall3solid
export fileformat=STEP
export numlevels=1
export numrefine=1
export maxh=1000.0 
export minh=0.0
export input_path=/vol8/home/hnu_lhz/hjy/NETGEN/test_code_part02/inputData/$filename\.$fileformat
export output_path=/vol8/home/hnu_lhz/hjy/NETGEN/result/$filename\_r$numrefine\_l$numlevels\_$core_n\_max$maxh\_min$minh\/

yhrun -p mt_test --mpi=pmix ./mesh_occ_mpi -i $input_path -o $output_path -l $numlevels -r $numrefine --maxh $maxh --minh $minh -v -adj