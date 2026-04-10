#pragma once
#include "parallelMeshData.h"
#include "metis.h"
#include <filesystem>


bool NewSubmesh(void* mesh, void* submesh);

idx_t* PartitionMesh(void *mesh, int parts);
int FindSurfElem(void *mesh, int *elem, surfMap_t &surfMap);
int ExtractFaceOutward(int *tverts, int *fverts);
void GetSurfPoints(void *mesh, surfMap_t &surfMap);
void ExtractSurfaceMesh(void *mesh, int fid, int procid, int *tverts, fid_xdMeshFaceInfo *sub_face_map, int sub_face_map_index, int domainidx);
void ExtractPartitionSurfaceMesh(void *mesh, idx_t *edest, std::map< int, xdMeshFaceInfo > &facemap);
void Allgather_Face_Map(std::map<int, xdMeshFaceInfo> &facemap, fid_xdMeshFaceInfo *sub_face_map, int ne, int sub_ne, int *every_sub_ne, int *every_offset);
void PartFaceCreate(void *mesh, int belongNumberPartition, std::map< int, xdMeshFaceInfo > &facemap, int maxbarycoord, void *submesh, std::map< int, int > &g2lvrtxmap, std::map< Barycentric, int, CompBarycentric > &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::list<xdFace> &newfaces);
void DumpFacesToFileAndClear(std::list<xdFace> &newfaces, const std::string &filepath);
void RefineOneLevelToFile(void *submesh, const std::string &infile, const std::string &outfile, std::size_t batch_faces, std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::map<IntPair, int, IntPairCompare> &edgemap);
void AddFacesFromFileToMesh(void *submesh, const std::string &filepath, std::size_t batch_faces);
void Refine_Stream(void *submesh, int numlevels, int belongNumberPartition, const std::string &stream_dir, std::size_t batch_faces, bool keep_stream_files, std::list<xdFace> &newfaces, std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::map<IntPair, int, IntPairCompare> &edgemap);
void Refine(void *submesh, int numlevels, int belongNumberPartition, std::list<xdFace> &newfaces, std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::map<IntPair, int, IntPairCompare> &edgemap);
void BarycMidPoint(Barycentric p1, Barycentric p2, Barycentric &res);
Barycentric InitBarycv(int v, int maxbarycoord);
void DumpCurrentSurfaceElementsToFile(void *submesh, const std::string &filepath, const std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap, const std::map<int, Barycentric> &locvrtx2barycmap);
void DumpCurrentVolumeElementsToFile(void *submesh, const std::string &filepath);
void DumpInitialPointsToPointTable(void *submesh, const std::string &path);
void ReplayNewPointsFromPointTableToMesh(void *submesh, const std::string &point_table_path, int start_point_id);
void Refineforvol_Stream(void *submesh, const std::string &surface_infile, const std::string &tet_infile, const std::string &surface_outfile, const std::string &tet_outfile, const std::string &point_table_path, std::size_t batch_faces, std::size_t batch_tets, int &next_point_id, std::size_t edge_shards, int round, std::map< Barycentric, int, CompBarycentric > &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::map < IntPair, int, IntPairCompare> &edgemap);
void *ReplayTetsFromFileToMesh(void *submesh, const std::string &filepath, std::size_t batch_tets);
void Refineforvol(void *submesh, int belongNumberPartition, std::list<xdFace> &newfaces, std::map< Barycentric, int, CompBarycentric > &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::map < IntPair, int, IntPairCompare> &edgemap);
void computeadj(int mypid, std::map< int, xdMeshFaceInfo > &facemap, std::map<int, int > &g2lvrtxmap, std::map< Barycvrtx, std::list<int>, CompBarycvrtx > &barycvrtx2adjprocsmap);
int* com_barycoords(void *submesh,MPI_Comm comm,std::map< Barycvrtx, std::list<int>, CompBarycvrtx > &barycvrtx2adjprocsmap,std::map< Barycentric, int, CompBarycentric > &baryc2locvrtxmap,std::map< int, std::list<int>> &adjbarycs,	int numprocs, int *newgVEid,int mypid);
int* com_baryVolumeElements(void *submesh, MPI_Comm comm, std::map< Barycvrtx, std::list<int>, CompBarycvrtx > &barycvrtx2adjprocsmap, std::map< Barycentric, int, CompBarycentric > &baryc2locvrtxmap, std::map< int, std::list<int>> &adjbarycs, int *oldgid, int *VEgid, std::list<VEindex> &VEindexs, int numprocs, int mypid);
// int* com_baryVolumeElements(void *submesh, MPI_Comm comm, std::map< Barycvrtx, std::list<int>, CompBarycvrtx > &barycvrtx2adjprocsmap, std::map< Barycentric, int, CompBarycentric > &baryc2locvrtxmap, std::map< int, std::list<int>> &adjbarycs, int *oldgid, int *VEgid,  int numprocs, int mypid);
void Record_LWR_count(double LWR, int * count);
bool meshQualityEvaluation(void *mesh, int id, std::string OUTPUT_PATH);
double triangle_jacobian_ratio(const double points[][3]);
double tetrahedrons_jacobian_ratio(const double points[][3]);
double getLenght(const double a1[], const double a2[]);
double length_width_ratio(const double points[][3]);
double tetrahedrons_length_width_ratio(const double points[][3]);
double min_internal_angle(const double points[][3]);
double max_internal_angle(const double points[][3]);
double func(const double a1[], const double a2[], const double a3[]);
double triangle_skew(const double points[][3]);
double tetrahedrons_skew(double points[][3]);
