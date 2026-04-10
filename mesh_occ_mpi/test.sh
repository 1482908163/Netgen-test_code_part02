#!/bin/bash

export core_n=32
export filename=wholewall3solid
export fileformat=STEP
export numlevels=1
export numrefine=1
export maxh=1000.0 
export minh=0.0
export input_path=/vol8/home/hnu_lhz/pwr/NETGEN/test_code_part02/inputData/$filename\.$fileformat
export output_path=/vol8/home/hnu_lhz/pwr/NETGEN/result/$filename\_r$numrefine\_l$numlevels\_$core_n\_max$maxh\_min$minh\/

mpicc --version
mpirun --version

module list 

#mpirun -disable-auto-cleanup -np $core_n ./mesh_occ_mpi -i $input_path -o $output_path -l $numlevels -r $numrefine --maxh $maxh --minh $minh -v -adj

yhrun -p mt_module -N 12 -n 48 /vol8/home/hnu_lhz/pwr/NETGEN/test_code_part02/build/mesh_occ_mpi/mesh_occ_mpi -i $input_path -o $output_path -l $numlevels -r $numrefine --maxh $maxh --minh $minh -v -adj