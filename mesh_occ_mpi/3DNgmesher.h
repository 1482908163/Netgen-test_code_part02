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

struct StreamMeshView {
    std::string point_table_path;
    std::string surface_file_path;
    std::string tet_file_path;
    int np = 0;
    int nse = 0;
    int ne = 0;
};

struct FaceKey
{
    int v[3];

    bool operator<(const FaceKey &other) const
    {
        if (v[0] != other.v[0]) return v[0] < other.v[0];
        if (v[1] != other.v[1]) return v[1] < other.v[1];
        return v[2] < other.v[2];
    }
};

inline FaceKey make_face_key(int a, int b, int c)
{
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    FaceKey key{{a, b, c}};
    return key;
}

struct StreamBoundaryHeaderStats {
    int boundary_count = 0;
};

struct StreamMeshQualityStats {
    int ne = 0;
    double min_volume = 0.0;
    double max_volume = 0.0;
    double avg_volume = 0.0;
};

struct StreamGhostVE {
    int gid;
    int domidx;
    int pindex[4];
    double xyz[4][3];
};

struct StreamVolWithAdjData {
    std::vector<PointCoordRecord> points;
    std::vector<int> point_global_ids;
    std::vector<TetRecord> local_tets;
    std::vector<TetRecord> ghost_tets;
    std::vector<FaceRecord> surfaces;
};

static_assert(std::is_trivially_copyable<StreamGhostVE>::value, "StreamGhostVE must be trivially copyable");

bool InitStreamMeshView(StreamMeshView &view,
    const std::string &point_table_path,
    const std::string &surface_file_path,
    const std::string &tet_file_path);
int StreamMesh_GetNP(const StreamMeshView &view);
int StreamMesh_GetNSE(const StreamMeshView &view);
int StreamMesh_GetNE(const StreamMeshView &view);
void StreamMesh_GetPoint(const StreamMeshView &view, int pid, double xyz[3]);
void StreamMesh_GetSurfaceElement(const StreamMeshView &view, int sid, int tri[3], int &surfidx);
void StreamMesh_GetVolumeElement(const StreamMeshView &view, int eid, int tet[4], int &domidx);
void DumpAdjBarycsSummary(const std::string &filepath,
    int mypid,
    const std::map<Barycvrtx, std::list<int>, CompBarycvrtx> &barycvrtx2adjprocsmap);
int *com_barycoords_from_streams(const StreamMeshView &smv,
    MPI_Comm comm,
    const std::map<Barycvrtx, std::list<int>, CompBarycvrtx> &barycvrtx2adjprocsmap,
    const std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
    std::map<int, std::list<int>> &adjbarycs,
    int numParts,
    int *VEgid,
    int id);
void WritePartitionNodesElementsFromStreams(const StreamMeshView &smv,
    const std::string &output_path,
    int numParts,
    int id,
    const int *newid,
    const int *VEgid);
void WritePartitionSharedFromAdjBarycs(const std::string &output_path,
    int numParts,
    int id,
    const int *newid,
    const std::map<int, std::list<int>> &adjbarycs);
void CollectPatboundFacesFromSurfaceFile(const std::string &surface_file_path,
    std::set<FaceKey> &patbound_faces);
StreamBoundaryHeaderStats WritePartitionBoundaryAndHeaderFromStreams(
    const StreamMeshView &smv,
    const std::string &output_path,
    int numParts,
    int id,
    const int *newid,
    const int *VEgid,
    const std::map<int, std::list<int>> &adjbarycs);
StreamMeshQualityStats ComputeMeshQualityFromStreams(const StreamMeshView &smv);
void WriteMeshQualityFromStreams(const std::string &output_path,
    int id,
    const StreamMeshQualityStats &stats);
void com_baryVolumeElements_from_streams(
    const StreamMeshView &smv,
    MPI_Comm comm,
    const std::map<int, std::list<int>> &adjbarycs,
    const int *newid,
    const int *VEgid,
    int numprocs,
    int mypid,
    StreamVolWithAdjData &out_data);
void WriteVolWithAdjFromStreams(const std::string &output_path,
    int id,
    const StreamVolWithAdjData &data);

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
bool WriteVolFromStreams(const std::string &point_table_path, const std::string &surface_file, const std::string &tet_file, const std::string &out_vol_path);
void Refineforvol_Stream(void *submesh, const std::string &surface_infile, const std::string &tet_infile, const std::string &surface_outfile, const std::string &tet_outfile, const std::string &point_table_path, std::size_t batch_faces, std::size_t batch_tets, int &next_point_id, std::size_t edge_shards, int round, std::map< Barycentric, int, CompBarycentric > &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::map < IntPair, int, IntPairCompare> &edgemap);
void *ReplayTetsFromFileToMesh(void *submesh, const std::string &filepath, std::size_t batch_tets);
void Refineforvol(void *submesh, int belongNumberPartition, std::list<xdFace> &newfaces, std::map< Barycentric, int, CompBarycentric > &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::map < IntPair, int, IntPairCompare> &edgemap);
void computeadj(int mypid, std::map< int, xdMeshFaceInfo > &facemap, std::map<int, int > &g2lvrtxmap, std::map< Barycvrtx, std::list<int>, CompBarycvrtx > &barycvrtx2adjprocsmap);
int com_sr_datatype(MPI_Comm comm, int num_s, int num_r, int *dest, int *src, int *s_length, int *r_length, Barycentric **s_data, Barycentric **r_data, MPI_Datatype datatype, int mypid);
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
