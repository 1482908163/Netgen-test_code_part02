ADD_EXECUTABLE(mesh_occ_mpi occmesh_MPI_main.cpp 3DNgmesher.cpp)

TARGET_LINK_LIBRARIES(mesh_occ_mpi libnglib.so)

TARGET_LINK_LIBRARIES(mesh_occ_mpi ${OCC_LIBRARIES})

TARGET_LINK_LIBRARIES(mesh_occ_mpi TKernel)

TARGET_LINK_LIBRARIES(mesh_occ_mpi metis)

TARGET_LINK_LIBRARIES(mesh_occ_mpi  /vol8/home/hnu_lhz/cjz/NETGEN/install/lib/libGKlib.a)


INSTALL(TARGETS mesh_occ_mpi   
       RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
