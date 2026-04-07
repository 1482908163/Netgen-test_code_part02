ADD_EXECUTABLE(merge_elmer merge_elmer.cpp)
TARGET_LINK_LIBRARIES(merge_elmer libnglib.so)

TARGET_LINK_LIBRARIES(merge_elmer ${OCC_LIBRARIES})

TARGET_LINK_LIBRARIES(merge_elmer TKernel)

TARGET_LINK_LIBRARIES(merge_elmer metis)

TARGET_LINK_LIBRARIES(merge_elmer /vol8/home/hnu_lhz/cjz/NETGEN/install/lib/libGKlib.a)



INSTALL(TARGETS    merge_elmer
       RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)