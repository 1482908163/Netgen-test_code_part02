SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)
PROJECT(HPDS_test)

SET(CMAKE_INSTALL_PREFIX /vol8/home/hnu_lhz/cjz/NETGEN/install)

add_definitions("-g")
add_definitions("-O3")
set(CMAKE_CXX_STANDARD 17)

INCLUDE_DIRECTORIES(NG_INCLUDE_DIR nglib.h /vol8/home/hnu_lhz/cjz/NETGEN/install/include)

FIND_PATH(NG_LIBRARY_DIR libnglib.so /vol8/home/hnu_lhz/cjz/NETGEN/install/lib)

SET(OpenCascade_DIR /vol8/home/hnu_lhz/cjz/NETGEN/install/lib/cmake/opencascade)
find_package(OpenCascade NAMES OpenCASCADE opencascade  ${OpenCascade_DIR} REQUIRED)
add_definitions(-DOCCGEOMETRY)
    set(OCC_LIBRARIES
      TKBO
      TKBRep
      TKBool
      TKCAF
      TKCDF
      TKFillet
      TKG2d
      TKG3d
      TKGeomAlgo
      TKGeomBase
      TKHLR
      TKIGES
      TKLCAF
      TKMath
      TKMesh
      TKOffset
      TKPrim
      TKSTEP
      TKSTEP209
      TKSTEPAttr
      TKSTEPBase
      TKSTL
      TKService
      TKShHealing
      TKTopAlgo
      TKV3d
      TKVCAF
      TKXCAF
      TKXDEIGES
      TKXDESTEP
      TKXSBase
      TKernel
    )

SET(OCC_LIBRARIES_DIR   /vol8/home/hnu_lhz/cjz/NETGEN/install/lib)
SET(OCC_INLUCDE_DIR     /vol8/home/hnu_lhz/cjz/NETGEN/install/include)

SET(METIS_LIBRARIES_DIR /vol8/home/hnu_lhz/cjz/NETGEN/install/lib)
SET(METIS_INCLUDE_DIR    /vol8/home/hnu_lhz/cjz/NETGEN/install/include)

# SET(MPI_LIBRARIES_DIR   /usr/lib/aarch64-linux-gnu/openmpi/lib/)
# SET(MPI_INCLUDE_DIR     /usr/lib/aarch64-linux-gnu/openmpi/include/)

LINK_LIBRARIES(${NG_LIBRARY_DIR} ${OCC_LIBRARY_DIR} ${METIS_LIBRARIES_DIR} ${MPI_LIBRARIES_DIR})
LINK_DIRECTORIES(${NG_LIBRARY_DIR} ${OCC_LIBRARY_DIR} ${METIS_LIBRARIES_DIR} ${MPI_LIBRARIES_DIR})
INCLUDE_DIRECTORIES(${NG_INCLUDE_DIR}  ${OpenCASCADE_INCLUDE_DIR} ${METIS_INCLUDE_DIR} ${MPI_INCLUDE_DIR})


ADD_SUBDIRECTORY(mesh_occ_mpi )
ADD_SUBDIRECTORY(merge_mesh )

