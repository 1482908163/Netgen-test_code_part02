#include <iostream>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include "mpi.h"
#include "TopTools_IndexedMapOfShape.hxx"
#include "TopoDS.hxx"
#include "TopoDS_Face.hxx"
#include "TopoDS_Shape.hxx"
#include "GProp_GProps.hxx"
#include "BRepGProp.hxx"
#include "3DNgmesher.h"
#include <filesystem>
#include <set>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

namespace nglib {
#include <nglib.h>
}
#include "createElmerOutput.h"

nglib::Ng_Mesh *ClearVolumeElementsOrEquivalent(nglib::Ng_Mesh *mesh);

static double get_current_rss_mb()
{
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
    {
        if (line.rfind("VmRSS:", 0) == 0)
        {
            std::istringstream iss(line);
            std::string key;
            double rss_kb = 0.0;
            std::string unit;
            iss >> key >> rss_kb >> unit;
            return rss_kb / 1024.0;
        }
    }
    return 0.0;
}

static double get_peak_rss_mb()
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024.0;
}

static void print_stage_mem(const char *tag, int id)
{
    double local_current = get_current_rss_mb();
    double local_peak = get_peak_rss_mb();
    double global_current = 0.0;
    double global_peak = 0.0;
    MPI_Reduce(&local_current, &global_current, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_peak, &global_peak, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (id == 0)
        std::cout << "[MEM] " << tag << " : current=" << global_current
                  << " MB, peak=" << global_peak << " MB" << std::endl;
}

//打印帮助信息
void print_help() {
    cerr << "参数输入帮助" << endl <<
         "-o : 输出文件目录 默认为./output" << endl <<
         "-i : 输入文件目录 默认为/work/home/moussa/wholewall3.stp" << endl <<
         "-l --levels : 细化等级 默认为0" << endl <<
         "-r --refine : 细化次数 默认为0" << endl <<
         "-maxh : 网格最大值 默认为1000.0" << endl <<
         "-minh : 网格最小值 默认为10.0" << endl <<
         "-v : 保存细化文件" << endl <<
         "--stream : 开启完整体流式细化路径" << endl <<
         "--stream-batch : 表面流批大小 默认为4096" << endl <<
         "--stream-vol-batch : 体流批大小 默认为4096" << endl <<
         "--stream-dir : 流式中间文件目录 默认为 <output>/stream" << endl <<
         "--stream-final-mode : 最终阶段模式 replay/file_only，默认为 replay" << endl <<
         "--keep-stream-files : 保留流式中间文件" << endl <<
         "-adj : 通信" << endl <<
         "-h --help : 参数输入帮助" << endl;
}

int main(int argc, char **argv) {

    using namespace nglib;

    int id; //进程号
    int p = 1;  //进程总数
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &id); //获取进程号
    MPI_Comm_size(MPI_COMM_WORLD, &p);  //获取进程总数

    // if(id == 0) cout << "MPI:" << p << endl;


    Ng_Init();

    Ng_Mesh *occ_mesh;

//Parameters
    string OUTPUT_PATH = "./output/";
    string INPUT_PATH = "/work/home/moussa/wholewall3.stp";
    bool save_vol = false;
    bool isComputeAdj = false;
    int numlevels = 0;
    int numrefine = 0;
    double maxh = 1000.0,minh = 10.0;
    bool stream_mode = false;
    std::size_t stream_batch = 4096;
    std::size_t stream_vol_batch = 4096;
    std::string stream_dir;
    std::string stream_final_mode = "replay";
    bool keep_stream_files = false;
//

    if(argc <= 1) {
        print_help();
        MPI_Finalize();
        return 1;
    }

    for(int i = 1; i < argc; i++) {
        if(!strcmp(argv[i],"-o")) {
            if(argv[i+1] != NULL) OUTPUT_PATH = argv[i+1];
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"-i")) {
            if(argv[i+1] != NULL) INPUT_PATH = argv[i+1];
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"-l") || !strcmp(argv[i],"--levels")) {
            if(argv[i+1] != NULL) numlevels = atoi(argv[i+1]);
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"-r") || !strcmp(argv[i],"--refine")) {
            if(argv[i+1] != NULL) {
                numrefine = atoi(argv[i+1]);
            }
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"--maxh")) {
            if(argv[i+1] != NULL) maxh = atof(argv[i+1]);
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"--minh")) {
            if(argv[i+1] != NULL) minh = atof(argv[i+1]);
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"-h") || !strcmp(argv[i],"--help")) {
            print_help();
            MPI_Finalize();
            return 1;
        }
        else if(!strcmp(argv[i],"-v")) {
            save_vol = true;
        }
        else if(!strcmp(argv[i],"--stream")) {
            stream_mode = true;
        }
        else if(!strcmp(argv[i],"--stream-batch")) {
            if(argv[i+1] != NULL) stream_batch = std::stoull(argv[i+1]);
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"--stream-vol-batch")) {
            if(argv[i+1] != NULL) stream_vol_batch = std::stoull(argv[i+1]);
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"--stream-dir")) {
            if(argv[i+1] != NULL) stream_dir = argv[i+1];
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"--stream-final-mode")) {
            if(argv[i+1] != NULL) {
                stream_final_mode = argv[i+1];
                if(stream_final_mode != "replay" && stream_final_mode != "file_only") {
                    print_help();
                    MPI_Finalize();
                    return 1;
                }
            }
            else {
                print_help();
                MPI_Finalize();
                return 1;
            }
        }
        else if(!strcmp(argv[i],"--keep-stream-files")) {
            keep_stream_files = true;
        }
        else if(!strcmp(argv[i],"-adj")) {
            isComputeAdj = true;

        }
    }

    if(stream_dir.empty()) {
        stream_dir = OUTPUT_PATH + "stream";
    }

    string savepvname = OUTPUT_PATH + "test_occ/test_occ.vol";

    if(id == 0) {

        cerr << "========参数========" << endl <<
             "使用核数 : " << p << endl <<
             "输出文件目录 : " << OUTPUT_PATH << endl <<
             "输入文件目录 : " << INPUT_PATH << endl <<
             "细化等级 : " << numlevels << endl <<
             "细化次数 : " << numrefine << endl <<
             "流式模式 : " << (stream_mode ? "on" : "off") << endl <<
             "表面流批大小 : " << stream_batch << endl <<
             "体流批大小 : " << stream_vol_batch << endl <<
             "流文件目录 : " << stream_dir << endl <<
             "最终阶段模式 : " << stream_final_mode << endl <<
             "网格最大值:" << maxh << endl <<
             "网格最小值:" << minh << endl;


        // int ret = mkdir(OUTPUT_PATH.c_str(), 0777);
        mkdir(OUTPUT_PATH.c_str(), 0777);

        string meshQuality_path = OUTPUT_PATH + string("meshQuality/");
        // int meshQuality_ret = mkdir(meshQuality_path.c_str(), 0777);
        mkdir(meshQuality_path.c_str(), 0777);

        string refinedSurfmesh_path = OUTPUT_PATH + string("refinedSurfmesh/");
        // int refinedSurfmesh_ret = mkdir(refinedSurfmesh_path.c_str(), 0777);
        mkdir(refinedSurfmesh_path.c_str(), 0777);

        string test_occ_path = OUTPUT_PATH + string("test_occ/");
        // int test_occ_ret = mkdir(test_occ_path.c_str(), 0777);
        mkdir(test_occ_path.c_str(), 0777);

        string testout_path = OUTPUT_PATH + string("testout/");
        // int testout_ret = mkdir(testout_path.c_str(), 0777);
        mkdir(testout_path.c_str(), 0777);

        string volfined_path = OUTPUT_PATH + string("volfined/");
        // int volfined_ret = mkdir(volfined_path.c_str(), 0777);
        mkdir(volfined_path.c_str(), 0777);

        string volwithadj_path = OUTPUT_PATH + string("volwithadj/");
        // int volwithadj_ret = mkdir(volwithadj_path.c_str(), 0777);
        mkdir(volwithadj_path.c_str(), 0777);

        if(stream_mode) {
            std::filesystem::create_directories(stream_dir);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(stream_mode) {
        std::filesystem::create_directories(stream_dir);
    }

    // Define pointer to OCC Geometry
    Ng_OCC_Geometry *occ_geom;

    // Ng_Mesh *occ_mesh;

    Ng_Meshing_Parameters mp;

    TopTools_IndexedMapOfShape FMap;

    // Ng_OCC_TopTools_IndexedMapOfShape *occ_fmap = (Ng_OCC_TopTools_IndexedMapOfShape*)&FMap;

    // Result of Netgen Operations
    Ng_Result ng_res;

    // Initialise the Netgen Core library
    // Ng_Init();

    // Read in the OCC File
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    string STEP_PATH = INPUT_PATH;
    occ_geom = Ng_OCC_Load_STEP(STEP_PATH.c_str());
    if (!occ_geom)
    {
        cout << "Error reading in STEP File: " << STEP_PATH << endl;
        MPI_Finalize();
        return 1;
    }
    if(id == 0)
        cout << "Successfully loaded STEP File: " << STEP_PATH << endl;


    occ_mesh = Ng_NewMesh();

    mp.uselocalh = 1;
    mp.elementsperedge = 2.0;
    mp.elementspercurve = 2.0;
    mp.maxh = maxh;
    mp.minh = minh;
    mp.grading = 0.3;
    mp.closeedgeenable = 0;
    mp.closeedgefact = 1.0;
    mp.optsurfmeshenable = 0;


    if(id == 0) {
        cout << "Setting Local Mesh size....." << endl;
        cout << "OCC Mesh Pointer before call = " << occ_mesh << endl;
    }
    Ng_OCC_SetLocalMeshSize(occ_geom, occ_mesh, &mp);
    if(id == 0) {
        cout << "Local Mesh size successfully set....." << endl;
        cout << "OCC Mesh Pointer after call = " << occ_mesh << endl;
        cout << "Creating Edge Mesh....." << endl;
    }

    ng_res = Ng_OCC_GenerateEdgeMesh(occ_geom, occ_mesh, &mp);
    if (ng_res != NG_OK)
    {
        Ng_DeleteMesh(occ_mesh);
        cout << "Error creating Edge Mesh.... Aborting!!" << endl;
        MPI_Finalize();
        return 1;
    }
    else
    {
        if(id == 0) {
            cout << "Edge Mesh successfully created....." << endl;
            cout << "Number of points = " << Ng_GetNP(occ_mesh) << endl;
        }
    }

    id == 0 ? cout << "Creating Surface Mesh....." << endl: cout << "";

    ng_res = Ng_OCC_GenerateSurfaceMesh(occ_geom, occ_mesh, &mp);
    if (ng_res != NG_OK)
    {
        Ng_DeleteMesh(occ_mesh);    //删除体网格
        cout << "Error creating Surface Mesh..... Aborting!!" << endl;
        MPI_Finalize();
        return 1;
    }
    else
    {
        if(id == 0) {
            cout << "Surface Mesh successfully created....." << endl;
            cout << "Number of points = " << Ng_GetNP(occ_mesh) << endl;
            cout << "Number of surface elements = " << Ng_GetNSE(occ_mesh) << endl;
        }
    }

    if(id == 0)
        cout << "Creating Volume Mesh....." << endl;

    ng_res = Ng_GenerateVolumeMesh(occ_mesh, &mp);

    if(id == 0) {

        cout << "Volume Mesh successfully created....." << endl;
        cout << "Number of points = " << Ng_GetNP(occ_mesh) << endl;
        cout << "Number of volume elements = " << Ng_GetNE(occ_mesh) << endl;
    }


    // if(id == 0) cout << "Saving Mesh as VOL file....." << endl;
    int faceNum = nglib::Ng_GetNFD((nglib::Ng_Mesh*)occ_mesh);
    int x[4];
    for (int i = 1; i < faceNum; i++) {
        nglib::My_Ng_GetFaceDescriptor((nglib::Ng_Mesh*)occ_mesh, i, x);
        //std::cout <<"+++++++++++++++" << x[0] << "=" << x[1] << "=" << x[2] << "=" << x[3] << std::endl;

    }

    if(id == 0)
        Ng_SaveMesh(occ_mesh, savepvname.c_str());


    if(id == 0) cout << "Generate Coarse Mesh Done..." << endl;
    MPI_Barrier(MPI_COMM_WORLD);


    // double Coarse_endTime = clock();
    double Coarse_endTime = MPI_Wtime();
    double Coarse_Time = (double)(Coarse_endTime - startTime);



    if (p >= 1) {

        double time[5] = {0,0,0,0,0};
        //Set the number of partitions
        int numParts = p;
        FILE *fp;
        FILE *fp_time;
        //set the level of refinement
        // int optvolmeshenable = 0;
        // int optsteps_3d = 0;
        // double gradingp = 0.3;
        // char paramname[20];

        //maxbarycoord is the n secondary of 2
        int maxbarycoord = 1 << (numlevels + numrefine+1);
        //int maxbarycoord = 1 << (numlevels + numrefine);
        map< int, xdMeshFaceInfo > facemap;
        map< int, int > g2lvrtxmap;
        map< Barycentric, int, CompBarycentric > baryc2locvrtxmap;
        map< int, Barycentric > locvrtx2barycmap;
        map< int, list<int>> adjbarycs;
        list<xdFace> newfaces;
        list<VEindex> VEindexs;
        list<VEindex>::iterator VEi;
        std::map< IntPair, int, IntPairCompare > edgemap;
        std::set<FaceKey> patbound_faces;
        bool use_saved_patbound_faces = false;

        std::string str_id = std::to_string(id);

        nglib::Ng_Mesh * submesh = nglib::Ng_NewMesh();
        NewSubmesh(occ_mesh, submesh);

        idx_t *edest = PartitionMesh(occ_mesh, numParts);

        //MPI_Barrier(MPI_COMM_WORLD);
        double currtime0 = MPI_Wtime();
        time[0] = double(currtime0 - Coarse_endTime);//NewSubmesh + PartitionMesh 等细化前准备/分区

        ExtractPartitionSurfaceMesh(occ_mesh, edest, facemap);
        free(edest);
        //MPI_Barrier(MPI_COMM_WORLD);
        double currtime1 = MPI_Wtime();
        time[1] = double(currtime1 - currtime0);


        PartFaceCreate(occ_mesh, id, facemap, maxbarycoord, submesh, g2lvrtxmap, baryc2locvrtxmap, locvrtx2barycmap, newfaces);
        // savepvname = OUTPUT_PATH + "PartFaceCreate/PartFaceCreate" + str_id + ".vol";
        // if(save_vol) {
        // 	Ng_SaveMesh(submesh, savepvname.c_str());
        // }

        //MPI_Barrier(MPI_COMM_WORLD);
        double currtime2 = MPI_Wtime();
        time[2] = double(currtime2 - currtime1);
        print_stage_mem("after PartFaceCreate", id);

	        bool skip_final_mesh_postprocess = false;
	        if(!stream_mode) {
            Refine(submesh, numlevels, id, newfaces, baryc2locvrtxmap, locvrtx2barycmap, edgemap);
        }
        else {
            Refine_Stream(submesh, numlevels, id, stream_dir, stream_batch, keep_stream_files, newfaces, baryc2locvrtxmap, locvrtx2barycmap, edgemap);
        }
        //MPI_Barrier(MPI_COMM_WORLD);
        double currtime3 = MPI_Wtime();
        time[3] = double(currtime3 - currtime2);
        print_stage_mem("after Refine_or_Stream", id);

        savepvname = OUTPUT_PATH + "refinedSurfmesh/refinedSurfmesh" + str_id + ".vol";
        if(save_vol) {
            Ng_SaveMesh(submesh, savepvname.c_str());
        }

        //The face mesh grid is refined in parallel to each partition.
        int i;
        nglib::Ng_Meshing_Parameters nmp;
        //nmp.maxh = 1e6;
        nmp.fineness = 1;

        double volumeMesh_start = MPI_Wtime();
        nglib::Ng_GenerateVolumeMesh(submesh, &nmp);
        double volumeMesh_end = MPI_Wtime();
        //MPI_Barrier(MPI_COMM_WORLD);
        if(id == 0) printf("meshing done \n");
        double currtime4 = MPI_Wtime();
        time[4] = double(currtime4 - currtime3);
        print_stage_mem("after Ng_GenerateVolumeMesh", id);

        if(!stream_mode) {
            // 原版非流式路径：surface refine 后保留 newfaces，
            // 每轮 volume refine 都直接在当前 submesh 上继续做。
            for (i = 0; i < numrefine; i++) {
                std::map< IntPair, int, IntPairCompare >().swap(edgemap);
                Refineforvol(submesh, id, newfaces, baryc2locvrtxmap, locvrtx2barycmap, edgemap);
            }
        }
	        else if(numrefine > 0) {
            // 完整体流式路径：
            // 1. 从当前 submesh 导出 surface/tet stream；
            // 2. 每轮执行 stream -> stream 的体细化；
            // 3. 最后一轮结束后再把最终 surface/tet 回灌到 submesh。
            //
            // 这样中间轮 refined tetra 不会长时间挂在 mesh 里，从而压低峰值内存。
	            std::string surface_in = stream_dir + "/surf_rank" + str_id + "_round0.bin";
	            std::string tet_in = stream_dir + "/tet_rank" + str_id + "_round0.bin";
	            std::string surface_out = stream_dir + "/surf_rank" + str_id + "_round1.bin";
	            std::string tet_out = stream_dir + "/tet_rank" + str_id + "_round1.bin";
	            std::string point_table_path = stream_dir + "/points_rank" + str_id + ".bin";
	            DumpCurrentSurfaceElementsToFile(submesh, surface_in, baryc2locvrtxmap, locvrtx2barycmap);
	            DumpInitialPointsToPointTable(submesh, point_table_path);
	            int initial_np = nglib::Ng_GetNP(submesh);
	            int next_point_id = initial_np + 1;
	            DumpCurrentVolumeElementsToFile(submesh, tet_in);
	            print_stage_mem("after DumpCurrentVolumeElementsToFile", id);
	            nglib::ClearSurfaceElements(submesh);
	            submesh = ClearVolumeElementsOrEquivalent(submesh);
	            print_stage_mem("after ClearVolumeElements", id);

	            for (i = 0; i < numrefine; i++) {
	                Refineforvol_Stream(submesh, surface_in, tet_in, surface_out, tet_out, point_table_path, stream_batch, stream_vol_batch, next_point_id, 16, i + 1, baryc2locvrtxmap, locvrtx2barycmap, edgemap);
	                if(!keep_stream_files) {
	                    std::filesystem::remove(surface_in);
	                    std::filesystem::remove(tet_in);
	                }
                std::swap(surface_in, surface_out);
                std::swap(tet_in, tet_out);
                surface_out = stream_dir + "/surf_rank" + str_id + "_round" + std::to_string(i + 2) + ".bin";
                tet_out = stream_dir + "/tet_rank" + str_id + "_round" + std::to_string(i + 2) + ".bin";
                print_stage_mem("after one Refineforvol_Stream round", id);
	            }

		            print_stage_mem("before final replay/postprocess", id);
		            if(stream_final_mode == "file_only") {
		                StreamMeshView smv;
		                InitStreamMeshView(smv, point_table_path, surface_in, tet_in);
		                const int stream_np = StreamMesh_GetNP(smv);
		                const int stream_nse = StreamMesh_GetNSE(smv);
		                const int stream_ne = StreamMesh_GetNE(smv);
		                double p1_xyz[3] = {0.0, 0.0, 0.0};
		                int s1_tri[3] = {0, 0, 0};
		                int s1_surfidx = 0;
		                int e1_tet[4] = {0, 0, 0, 0};
		                int e1_domidx = 0;
		                StreamMesh_GetPoint(smv, 1, p1_xyz);
		                StreamMesh_GetSurfaceElement(smv, 1, s1_tri, s1_surfidx);
		                StreamMesh_GetVolumeElement(smv, 1, e1_tet, e1_domidx);
		                const std::string streammesh_dump_path = OUTPUT_PATH + "testout/testout_streammesh.txt";
		                std::map<Barycvrtx, std::list<int>, CompBarycvrtx> barycvrtx2adjprocsmap_stream;
		                print_stage_mem("before computeadj(file_only)", id);
		                computeadj(id, facemap, g2lvrtxmap, barycvrtx2adjprocsmap_stream);
		                if(barycvrtx2adjprocsmap_stream.empty()) {
		                    std::cout << "[STREAM-WARN] computeadj produced empty adjacency map on rank " << id << std::endl;
		                }
		                std::filesystem::create_directories(OUTPUT_PATH + "testout");
		                const std::string streamadj_dump_path = OUTPUT_PATH + "testout/testout_streamadj.txt";
		                int *VEgid = nullptr;
		                MYCALLOC(VEgid, int *, (stream_ne + 1), sizeof(int));
		                std::map<int, std::list<int>> adjbarycs_stream;
		                int *newid = com_barycoords_from_streams(
		                    smv,
		                    MPI_COMM_WORLD,
		                    barycvrtx2adjprocsmap_stream,
		                    baryc2locvrtxmap,
		                    adjbarycs_stream,
		                    numParts,
		                    VEgid,
		                    id);
		                print_stage_mem("after com_barycoords_from_streams", id);
		                WritePartitionNodesElementsFromStreams(
		                    smv,
		                    OUTPUT_PATH,
		                    numParts,
		                    id,
		                    newid,
		                    VEgid);
		                WritePartitionSharedFromAdjBarycs(
		                    OUTPUT_PATH,
		                    numParts,
		                    id,
		                    newid,
		                    adjbarycs_stream);
		                const StreamBoundaryHeaderStats bh_stats =
		                    WritePartitionBoundaryAndHeaderFromStreams(
		                        smv,
		                        OUTPUT_PATH,
		                        numParts,
		                        id,
		                        newid,
		                        VEgid,
		                        adjbarycs_stream);
		                StreamVolWithAdjData voladj_data;
		                print_stage_mem("before com_baryVolumeElements_from_streams", id);
		                com_baryVolumeElements_from_streams(
		                    smv,
		                    MPI_COMM_WORLD,
		                    adjbarycs_stream,
		                    newid,
		                    VEgid,
		                    numParts,
		                    id,
		                    voladj_data);
		                print_stage_mem("after com_baryVolumeElements_from_streams", id);
		                const StreamFullMeshQualityStats mq_stats =
		                    ComputeFullMeshQualityFromVolWithAdjStreams(voladj_data, newid);
		                print_stage_mem("after ComputeFullMeshQualityFromVolWithAdjStreams", id);
		                WriteFullMeshQualityFromVolWithAdjStreams(
		                    OUTPUT_PATH,
		                    id,
		                    mq_stats,
		                    MPI_COMM_WORLD);
		                WriteVolWithAdjFromStreams(OUTPUT_PATH, id, voladj_data, newid);
		                print_stage_mem("after WriteVolWithAdjFromStreams", id);
		                const std::string stream_barycoords_dump_path = OUTPUT_PATH + "testout/testout_stream_barycoords.txt";
		                const std::string stream_shared_dump_path = OUTPUT_PATH + "testout/testout_stream_shared.txt";
		                const std::string stream_boundary_dump_path = OUTPUT_PATH + "testout/testout_stream_boundary.txt";
		                const std::string stream_meshquality_dump_path = OUTPUT_PATH + "testout/testout_stream_meshquality.txt";
		                const std::string stream_volwithadj_dump_path = OUTPUT_PATH + "testout/testout_stream_volwithadj.txt";
		                const std::string stream_newid_duplicates_dump_path = OUTPUT_PATH + "testout/testout_stream_newid_duplicates.txt";
		                const int first_newid = (stream_np >= 1) ? newid[1] : 0;
		                const int first_vegid = (stream_ne >= 1) ? VEgid[1] : 0;
		                const int local_points = stream_np;
		                const int local_tets = stream_ne;
		                const int ghost_added_points = static_cast<int>(voladj_data.ghost_points.size());
		                const int ghost_added_tets = static_cast<int>(voladj_data.ghost_tets.size());
		                const int final_points = local_points + ghost_added_points;
		                const int final_tets = local_tets + ghost_added_tets;
		                int shared_vertex_count = 0;
		                int owned_shared_vertex_count = 0;
		                int received_shared_vertex_count = 0;
		                for(const auto &entry : baryc2locvrtxmap) {
		                    Barycvrtx key;
		                    key.gvrtx[0] = entry.first.gvrtx[0];
		                    key.gvrtx[1] = entry.first.gvrtx[1];
		                    key.gvrtx[2] = entry.first.gvrtx[2];
		                    const auto adj_it = barycvrtx2adjprocsmap_stream.find(key);
		                    if(adj_it == barycvrtx2adjprocsmap_stream.end()) {
		                        continue;
		                    }
		                    ++shared_vertex_count;
		                    std::vector<int> holders;
		                    holders.push_back(id);
		                    holders.insert(holders.end(), adj_it->second.begin(), adj_it->second.end());
		                    std::sort(holders.begin(), holders.end());
		                    if(adj_it->second.empty()) {
		                        std::cerr << "[STREAM-ERR] empty adjacency list for shared barycentric vertex on rank " << id << std::endl;
		                        MPI_Abort(MPI_COMM_WORLD, 1);
		                        return 1;
		                    }
		                    const int owner_index =
		                        (entry.first.gvrtx[0] + entry.first.gvrtx[1] + entry.first.gvrtx[2] +
		                         entry.first.coord[0] + entry.first.coord[1] + entry.first.coord[2]) %
		                        static_cast<int>(adj_it->second.size());
		                    const int ownerpid = holders[owner_index];
		                    if(ownerpid == id) {
		                        ++owned_shared_vertex_count;
		                    }
		                    else {
		                        ++received_shared_vertex_count;
		                    }
		                }
		                std::set<int> unique_newids;
		                int minus_one_count = 0;
		                for(int local_vid = 1; local_vid <= stream_np; ++local_vid) {
		                    if(newid[local_vid] == -1) {
		                        ++minus_one_count;
		                    }
		                    else if(newid[local_vid] > 0) {
		                        unique_newids.insert(newid[local_vid]);
		                    }
		                }
		                int shared_line_count = 0;
		                std::vector<std::string> shared_samples;
		                shared_samples.reserve(5);
		                for(const auto &entry : adjbarycs_stream) {
		                    std::vector<int> parts;
		                    parts.push_back(id);
		                    parts.insert(parts.end(), entry.second.begin(), entry.second.end());
		                    std::sort(parts.begin(), parts.end());
		                    parts.erase(std::unique(parts.begin(), parts.end()), parts.end());
		                    if(parts.size() <= 1) {
		                        continue;
		                    }
		                    ++shared_line_count;
		                    if(shared_samples.size() < 5) {
		                        std::ostringstream sample;
		                        sample << "item" << shared_samples.size() << "=" << newid[entry.first] << " -> ";
		                        for(std::size_t part_idx = 0; part_idx < parts.size(); ++part_idx) {
		                            sample << (parts[part_idx] + 1);
		                            if(part_idx + 1 < parts.size()) {
		                                sample << " ";
		                            }
		                        }
		                        shared_samples.push_back(sample.str());
		                    }
		                }
		                std::set<FaceKey> patbound_faces_stream;
		                CollectPatboundFacesFromSurfaceFile(smv.surface_file_path, patbound_faces_stream);
		                const std::filesystem::path boundary_file_path =
		                    std::filesystem::path(OUTPUT_PATH) /
		                    ("partitioning." + std::to_string(numParts)) /
		                    ("part." + std::to_string(id + 1) + ".boundary");
		                std::vector<std::string> boundary_samples;
		                boundary_samples.reserve(5);
		                {
		                    std::ifstream boundary_input(boundary_file_path);
		                    std::string boundary_line;
		                    while (std::getline(boundary_input, boundary_line) && boundary_samples.size() < 5) {
		                        if (!boundary_line.empty()) {
		                            std::ostringstream sample;
		                            sample << "item" << boundary_samples.size() << "=" << boundary_line;
		                            boundary_samples.push_back(sample.str());
		                        }
		                    }
		                }
		                if(id == 0) {
		                    std::ofstream reset_streammesh_dump(streammesh_dump_path, std::ios::trunc);
		                    std::ofstream reset_streamadj_dump(streamadj_dump_path, std::ios::trunc);
		                    std::ofstream reset_stream_barycoords_dump(stream_barycoords_dump_path, std::ios::trunc);
		                    std::ofstream reset_stream_shared_dump(stream_shared_dump_path, std::ios::trunc);
		                    std::ofstream reset_stream_boundary_dump(stream_boundary_dump_path, std::ios::trunc);
		                    std::ofstream reset_stream_meshquality_dump(stream_meshquality_dump_path, std::ios::trunc);
		                    std::ofstream reset_stream_volwithadj_dump(stream_volwithadj_dump_path, std::ios::trunc);
		                    std::ofstream reset_stream_newid_duplicates_dump(stream_newid_duplicates_dump_path, std::ios::trunc);
		                }
		                MPI_Barrier(MPI_COMM_WORLD);
		                for(int writer_rank = 0; writer_rank < p; ++writer_rank) {
		                    if(id == writer_rank) {
		                        std::ofstream streammesh_dump(streammesh_dump_path, std::ios::app);
		                        streammesh_dump << "rank=" << id << "\n";
		                        streammesh_dump << "stream np=" << stream_np << "\n";
		                        streammesh_dump << "stream nse=" << stream_nse << "\n";
		                        streammesh_dump << "stream ne=" << stream_ne << "\n";
		                        streammesh_dump << "p1=" << p1_xyz[0] << " " << p1_xyz[1] << " " << p1_xyz[2] << "\n";
		                        streammesh_dump << "s1=" << s1_tri[0] << " " << s1_tri[1] << " " << s1_tri[2] << " surfidx=" << s1_surfidx << "\n";
		                        streammesh_dump << "e1=" << e1_tet[0] << " " << e1_tet[1] << " " << e1_tet[2] << " " << e1_tet[3] << " domidx=" << e1_domidx << "\n";
		                        streammesh_dump << "\n";
		                        DumpAdjBarycsSummary(streamadj_dump_path, id, barycvrtx2adjprocsmap_stream);
		                        std::ofstream stream_barycoords_dump(stream_barycoords_dump_path, std::ios::app);
		                        stream_barycoords_dump << "rank=" << id << "\n";
		                        stream_barycoords_dump << "np=" << stream_np << "\n";
		                        stream_barycoords_dump << "ne=" << stream_ne << "\n";
		                        stream_barycoords_dump << "shared_vertex_count=" << shared_vertex_count << "\n";
		                        stream_barycoords_dump << "owned_shared_vertex_count=" << owned_shared_vertex_count << "\n";
		                        stream_barycoords_dump << "received_shared_vertex_count=" << received_shared_vertex_count << "\n";
		                        stream_barycoords_dump << "newid[1]=" << first_newid << "\n";
		                        stream_barycoords_dump << "VEgid[1]=" << first_vegid << "\n";
		                        stream_barycoords_dump << "adjbarycs.size()=" << adjbarycs_stream.size() << "\n";
		                        stream_barycoords_dump << "\n";
		                        std::ofstream stream_shared_dump(stream_shared_dump_path, std::ios::app);
		                        stream_shared_dump << "rank=" << id << "\n";
		                        stream_shared_dump << "adjbarycs.size()=" << adjbarycs_stream.size() << "\n";
		                        stream_shared_dump << "shared_line_count=" << shared_line_count << "\n";
		                        for(const std::string &sample : shared_samples) {
		                            stream_shared_dump << sample << "\n";
		                        }
		                        stream_shared_dump << "\n";
		                        std::ofstream stream_boundary_dump(stream_boundary_dump_path, std::ios::app);
		                        stream_boundary_dump << "rank=" << id << "\n";
		                        stream_boundary_dump << "np=" << stream_np << "\n";
		                        stream_boundary_dump << "ne=" << stream_ne << "\n";
		                        stream_boundary_dump << "nse=" << stream_nse << "\n";
		                        stream_boundary_dump << "patbound_faces.size()=" << patbound_faces_stream.size() << "\n";
		                        stream_boundary_dump << "boundary_count=" << bh_stats.boundary_count << "\n";
		                        for(const std::string &sample : boundary_samples) {
		                            stream_boundary_dump << sample << "\n";
		                        }
		                        stream_boundary_dump << "\n";
		                        std::ofstream stream_meshquality_dump(stream_meshquality_dump_path, std::ios::app);
		                        stream_meshquality_dump << std::setprecision(17);
		                        stream_meshquality_dump << "rank=" << id << "\n";
		                        stream_meshquality_dump << "ne=" << mq_stats.ne << "\n";
		                        stream_meshquality_dump << "min_volume=" << mq_stats.min_volume << "\n";
		                        stream_meshquality_dump << "max_volume=" << mq_stats.max_volume << "\n";
		                        stream_meshquality_dump << "avg_volume="
		                                               << ((mq_stats.ne == 0) ? 0.0 : (mq_stats.volume_sum / mq_stats.ne))
		                                               << "\n";
		                        stream_meshquality_dump << "\n";
		                        std::ofstream stream_volwithadj_dump(stream_volwithadj_dump_path, std::ios::app);
		                        stream_volwithadj_dump << "rank=" << id << "\n";
		                        stream_volwithadj_dump << "local_points=" << local_points << "\n";
		                        stream_volwithadj_dump << "ghost_added_points=" << ghost_added_points << "\n";
		                        stream_volwithadj_dump << "local_tets=" << local_tets << "\n";
		                        stream_volwithadj_dump << "ghost_added_tets=" << ghost_added_tets << "\n";
		                        stream_volwithadj_dump << "final_points=" << final_points << "\n";
		                        stream_volwithadj_dump << "final_tets=" << final_tets << "\n";
		                        stream_volwithadj_dump << "\n";
		                        std::ofstream stream_newid_duplicates_dump(stream_newid_duplicates_dump_path, std::ios::app);
		                        stream_newid_duplicates_dump << "rank=" << id << "\n";
		                        stream_newid_duplicates_dump << "local_np=" << stream_np << "\n";
		                        stream_newid_duplicates_dump << "unique_newid_count=" << unique_newids.size() << "\n";
		                        stream_newid_duplicates_dump << "minus_one_count=" << minus_one_count << "\n";
		                        stream_newid_duplicates_dump << "\n";
		                    }
		                    MPI_Barrier(MPI_COMM_WORLD);
		                }
		                if(id == 0) {
		                    std::cout << "[STREAM] computeadj in file_only mode finished" << std::endl;
		                    std::cout << "[STREAM] com_barycoords_from_streams finished" << std::endl;
		                    std::cout << "[STREAM] partition nodes/elements written from streams" << std::endl;
		                    std::cout << "[STREAM] partition shared written from streams" << std::endl;
		                    std::cout << "[STREAM] partition boundary/header written from streams" << std::endl;
		                    std::cout << "[STREAM] meshQuality written from streams" << std::endl;
		                    std::cout << "[STREAM] volwithadj written from streams" << std::endl;
		                    std::cout << "[STREAM] final mode=file_only, skip replay/postprocess" << std::endl;
		                }
		                if(save_vol) {
		                    const std::string stream_vol_path = OUTPUT_PATH + "volfined/volfined" + str_id + ".vol";
		                    if(!WriteVolFromStreams(point_table_path, surface_in, tet_in, stream_vol_path)) {
		                        std::cerr << "failed to write .vol from streams: " << stream_vol_path << std::endl;
		                        MPI_Abort(MPI_COMM_WORLD, 1);
		                        return 1;
		                    }
		                }
		                free(newid);
		                free(VEgid);
		                print_stage_mem("after final replay/postprocess", id);
		                skip_final_mesh_postprocess = true;
		            }
		            else {
		                CollectPatboundFacesFromSurfaceFile(surface_in, patbound_faces);
		                use_saved_patbound_faces = true;
		                ReplayNewPointsFromPointTableToMesh(submesh, point_table_path, initial_np + 1);
		                AddFacesFromFileToMesh(submesh, surface_in, stream_batch);
		                submesh = (Ng_Mesh *)ReplayTetsFromFileToMesh(submesh, tet_in, stream_vol_batch);
		                print_stage_mem("after ReplayTetsFromFileToMesh", id);
		                print_stage_mem("after final replay/postprocess", id);

		                if(!keep_stream_files) {
		                    std::filesystem::remove(surface_in);
		                    std::filesystem::remove(tet_in);
		                    std::filesystem::remove(point_table_path);
		                }
		            }
		        }

	        if(!skip_final_mesh_postprocess) {
	        std::string savepvname = OUTPUT_PATH + "volfined/volfined" + str_id + ".vol";

        if(save_vol) {
            nglib::Ng_SaveMesh(submesh, savepvname.c_str());
        }


        if (isComputeAdj) {
            map<Barycvrtx, list<int>, CompBarycvrtx> barycvrtx2adjprocsmap;
            computeadj(id,facemap,g2lvrtxmap, barycvrtx2adjprocsmap);

            int *VEgid;
            int numNEs = nglib::Ng_GetNE(submesh);
            MYCALLOC(VEgid, int *, (numNEs + 1), sizeof(int));

            printf("start com_barycoords, id: %d\n", id);
            // cout << id << "start com_barycoords" << endl;

            int *newid = com_barycoords(submesh, MPI_COMM_WORLD, barycvrtx2adjprocsmap,
                                        baryc2locvrtxmap, adjbarycs, numParts, VEgid, id);

            
            // int pointdebug = nglib::Ng_GetNP((nglib::Ng_Mesh *)submesh);
            // char *debugpath = new char[512];
            // sprintf(debugpath,"pointdebugpath%d.txt",id);
            // std::string pointdebugpath = OUTPUT_PATH + debugpath;
            // ofstream outpointdebugpath1(pointdebugpath.c_str());
            // for(int i=1;i<=pointdebug;i++){
            //     outpointdebugpath1 << newid[i] << endl;
            // }
            // outpointdebugpath1.close();


            //newid 本地点id--》全局点id
            //VEgid 本地体网格id --》全局体网格id
            //adjbarycs 共享点再哪些处理器上

            // try{
            //     createElmerOutput(submesh,VEgid,newid,adjbarycs,numParts,id,OUTPUT_PATH);
            // }catch(...){
            //     printf("createElmerOutput error id is %d\n", id);
            // }
#if 1
            nglib::Ng_Mesh* mesh = (nglib::Ng_Mesh *)submesh;
            char * boundaryfile1 = new char[512];
            char * elementfile1 = new char[512];
            char * headerfile1 = new char[512];
            char * nodefile1 = new char[512];
            char * sharedfile1 = new char[512];
            char * path1 = new char[512];

            sprintf(path1,"partitioning.%d",numParts);
            sprintf(boundaryfile1, "partitioning.%d/part.%d.boundary", numParts, id+1);
            sprintf(elementfile1, "partitioning.%d/part.%d.elements", numParts, id+1);
            sprintf(headerfile1, "partitioning.%d/part.%d.header", numParts, id+1);
            sprintf(nodefile1, "partitioning.%d/part.%d.nodes", numParts, id+1);
            sprintf(sharedfile1, "partitioning.%d/part.%d.shared", numParts, id+1);

            string path = OUTPUT_PATH + string(path1);
            string boundaryfile = OUTPUT_PATH + string(boundaryfile1);
            string elementfile = OUTPUT_PATH + string(elementfile1);
            string headerfile = OUTPUT_PATH + string(headerfile1);
            string nodefile = OUTPUT_PATH + string(nodefile1);
            string sharedfile = OUTPUT_PATH + string(sharedfile1);

            mkdir(path.c_str(),0777);

            int ne = nglib::Ng_GetNE(mesh); //体网格的数量
            int nse = nglib::Ng_GetNSE(mesh); //面网格的数量
            int np = nglib::Ng_GetNP(mesh); //点的数量

            //输出elements文件
            ofstream outelements(elementfile.c_str());
            int tet[4];
            for(int i=0;i < ne;i++){
                nglib::Ng_GetVolumeElement (mesh, i+1, tet);
                outelements << VEgid[i+1] << " 1 504 " << newid[tet[0]] << " " << newid[tet[1]] << " " << newid[tet[2]] << " " << newid[tet[3]] << endl;
            }
            outelements.close();

            //输出nodes文件
            ofstream outnodes(nodefile.c_str());
            double point[3];
            for(int i=0; i<np;i++){
                nglib::Ng_GetPoint (mesh, i+1, point);
                outnodes << newid[i+1] << " -1 " << point[0] << " " << point[1] << " " << point[2] << endl;
            }
            outnodes.close();

            //输出shared文件
            ofstream outshareds(sharedfile.c_str());
            for(auto it = adjbarycs.begin();it != adjbarycs.end();it++){
                int locid = it->first;
                std::list<int> proceid = it->second;
                int sizeid = proceid.size() + 1;
                std::string sharednode;
                sharednode += std::to_string(sizeid);
                sharednode += " ";
                sharednode += std::to_string(id+1); //elmerID = 核心ID + 1
                sharednode += " ";
                int m = 0;
                for(auto itt = proceid.begin(); itt != proceid.end() ; itt++){
                    sharednode += std::to_string(((*itt)+1));     //elmerID = 核心ID + 1
                    if( m != (sizeid-1) ){
                        sharednode += " ";
                    }
                    m++;
                }
                outshareds << newid[locid] << " " << sharednode << endl;
            }
            outshareds.close();


            //输出boundary文件
            ofstream outboundarys(boundaryfile.c_str());
            //求边界面网格所在的体网格
            Index3 i3;
            int l;
            bool (*fn_pt)(Index3,Index3) = fncomp;
            std::multimap<Index3,int, bool(*)(Index3, Index3)> face2vol(fn_pt);
            std::multimap<Index3,int, bool(*)(Index3, Index3)>::iterator myit;
            for(int i=1; i<=ne;i++){
                nglib::Ng_GetVolumeElement (mesh, i, tet);
                for (int j = 1; j <= 4; j++){
                    l = 0;
                    for (int k = 1; k <= 4; k++)
                    {
                        if (k != j)
                        {
                            i3.x[l] = newid[tet[k-1]];
                            l++;
                        }
                    }
                    i3.Sort();
                    face2vol.insert(pair<Index3,int>(i3,VEgid[i]));
                }
            }

            int *surfpointss = new int[3];
            int surfidx;
            int geoid;
            int number = 0;
            //int nfd = ((Mesh*)mesh)->GetNFD();
            for(int j=0; j< nse; j++){
                nglib::Ng_GetSurfaceElement(mesh, j + 1, surfpointss, surfidx);
                // if((Mesh*)mesh->GetFaceDesriptor(mesh->SurfaceElement(j).GetIndex()).BCProperty()==nfd){
                //     continue;
                // }
                geoid = nglib::GetBoundaryID(mesh,j+1) +1;
                FaceKey fk = make_face_key(surfpointss[0], surfpointss[1], surfpointss[2]);
                if(use_saved_patbound_faces ? patbound_faces.count(fk) != 0 : nglib::ispatbound(mesh,j+1)){
                    continue;
                }
                i3.x[0] = newid[surfpointss[0]];
                i3.x[1] = newid[surfpointss[1]];
                i3.x[2] = newid[surfpointss[2]];
                i3.Sort();
                myit = face2vol.find(i3);
                if(myit!= face2vol.end()){
                    number++;
                    outboundarys << number << " " << geoid << " " << myit->second << " " << "0" << " 303 " << newid[surfpointss[0]]  << " " << newid[surfpointss[1]] << " " << newid[surfpointss[2]] <<endl;
                }
            }
            outboundarys.close();

            //输出header文件
            ofstream outheader(headerfile.c_str());
            outheader << np << " " << ne << " " << number << endl;
            outheader << 2 << endl;
            outheader << "504 " << ne << endl;
            outheader << "303 " << number << endl;
            if(adjbarycs.size() != 0)
            {
                outheader << adjbarycs.size() << " 0" << endl;
            }
            outheader.close();


#endif

            printf("start com_baryVolumeElements, id: %d\n", id);
            com_baryVolumeElements(submesh, MPI_COMM_WORLD, barycvrtx2adjprocsmap,
                                   baryc2locvrtxmap, adjbarycs, newid, VEgid, VEindexs, numParts, id);
            printf("createElmerOutput, id: %d\n", id);
            

            savepvname = OUTPUT_PATH + "volwithadj/volwithadj" + str_id + ".vol";
            if(save_vol) {
                nglib::Ng_SaveMesh(submesh, savepvname.c_str());

                // string openfoampath = OUTPUT_PATH + "openfoam/part" + str_id;
                // mkdir(openfoampath.c_str(), 0777);
                // const std::filesystem::path  &outfile = openfoampath;
                // nglib::My_WriteOpenFOAMFormat(submesh,outfile);
            }


        }

        /*double endTime = MPI_Wtime();
        double Fine_Time = (double)(endTime - Coarse_endTime);
        double runtime = (double)(endTime - startTime);

        savepvname = OUTPUT_PATH + "volwithadj/volwithadj" + str_id + ".vol";
        if(save_vol) {

        nglib::Ng_SaveMesh(submesh, savepvname.c_str());
        }
        if(id == 0) {
            string savepvname_time = OUTPUT_PATH + "testout/testout_time" + str_id + ".txt";
            fp_time = fopen(savepvname_time.c_str(), "w");
            if (fp_time == NULL) {
                cout << "File " << savepvname << "canot open" << endl;
            }
            else {
            fprintf(fp_time, "Coarse_Time for id:%d is %.2f s\r\n", id, Coarse_Time);
            fprintf(fp_time, "Fine_Time for id:%d is %.2f s\r\n", id, Fine_Time);
            fprintf(fp_time, "runtime for id:%d is %.2f s\r\n", id, runtime);
            for(int i = 0; i < 5; i++) {
                fprintf(fp_time, "part %d time : %.2f \n", i, time[i]);
            }
            }
            fclose(fp_time);
        }

        savepvname = OUTPUT_PATH + "testout/testout_mesh.txt";
        fp = fopen(savepvname.c_str(), "a");
        if (fp == NULL) {
            cout << "File " << savepvname << "canot open" << endl;
        }
        else {
            //fprintf(fp, "the num of points for id:%d is %d\r\n", id, nglib::Ng_GetNP(submesh));
            //fprintf(fp, "the num of Surelemments for id:%d is %d\r\n", id, nglib::Ng_GetNSE(submesh));
            fprintf(fp, "the num of Volelements for id:%d is %d\r\n", id, nglib::Ng_GetNE(submesh));
            fprintf(fp, "the volmesh generate time for if: %d is %f\r\n", id, volumeMesh_end-volumeMesh_start);
        }
        if(id == 0) {
            int Volelements_Sum = nglib::Ng_GetNE(submesh);
            for(int i = 1; i < p; i++) {
                int Volelements_Buf = 0;
                MPI_Recv(&Volelements_Buf, sizeof(Volelements_Buf), MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                Volelements_Sum += Volelements_Buf;
            }
            fprintf(fp, "the Sum of Volelements id %d\r\n", Volelements_Sum);

        }
        else {
            int Volelements_Buf = nglib::Ng_GetNE(submesh);
            MPI_Send(&Volelements_Buf, sizeof(Volelements_Buf), MPI_INT, 0, 0, MPI_COMM_WORLD);
        }


        fclose(fp);		/*
        int *VEgids_list, *VEgid_isin_list;
        MYCALLOC(VEgids_list, int *, (VEindexs.size() + 1), sizeof(int));
        MYCALLOC(VEgid_isin_list, int *, (VEindexs.size() + 1), sizeof(int));
        i = 1;
        for (VEi = VEindexs.begin(); VEi != VEindexs.end(); ++VEi) {
            VEgids_list[i] = (*VEi).gid;
            VEgid_isin_list[i] = (*VEi).Isin;
            i++;
        }
        */
        //}
        //else {
        double endTime = MPI_Wtime();
        double Fine_Time = (double)(endTime - Coarse_endTime);
        double runtime = (double)(endTime - startTime);
        savepvname = OUTPUT_PATH + "testout/testout_mesh.txt";
        if(id == 0) {
            string savepvname_time = OUTPUT_PATH + "testout/testout_time" + str_id + ".txt";
            fp_time = fopen(savepvname_time.c_str(), "w");
            if (fp_time == NULL) {
                cout << "File " << savepvname << "canot open" << endl;
            }
            else {
                fprintf(fp_time, "Coarse_Time for id:%d is %.2f s\r\n", id, Coarse_Time);
                fprintf(fp_time, "Fine_Time for id:%d is %.2f s\r\n", id, Fine_Time);
                fprintf(fp_time, "runtime for id:%d is %.2f s\r\n", id, runtime);
                for(int i = 0; i < 5; i++) {
                    fprintf(fp_time, "part %d time : %.2f \n", i, time[i]);
                }
            }
            fclose(fp_time);
        }
        // savepvname = OUTPUT_PATH + "testout/testout.txt";

        fp = fopen(savepvname.c_str(), "a");
        if (fp == NULL) {
            cout << "File " << savepvname << "canot open" << endl;
        }
        else {
            //fprintf(fp, "the num of points for id:%d is %d\r\n", id, nglib::Ng_GetNP(submesh));
            //fprintf(fp, "the num of Surelemments for id:%d is %d\r\n", id, nglib::Ng_GetNSE(submesh));
            fprintf(fp, "the num of Volelements for id:%d is %d\r\n", id, nglib::Ng_GetNE(submesh));
            fprintf(fp, "the volmesh generate time for if: %d is %f\r\n", id, volumeMesh_end-volumeMesh_start);
        }
        if(id == 0) {
            int Volelements_Sum = nglib::Ng_GetNE(submesh);
            for(int i = 1; i < p; i++) {
                int Volelements_Buf = 0;
                MPI_Recv(&Volelements_Buf, sizeof(Volelements_Buf), MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                Volelements_Sum += Volelements_Buf;
            }
            fprintf(fp, "the Sum of Volelements id %d\r\n", Volelements_Sum);

        }
        else {
            int Volelements_Buf = nglib::Ng_GetNE(submesh);
            MPI_Send(&Volelements_Buf, sizeof(Volelements_Buf), MPI_INT, 0, 0, MPI_COMM_WORLD);
        }


	        fclose(fp);
	        //}
	        meshQualityEvaluation(submesh, id, OUTPUT_PATH);
	        }


	    }

    if(id == 0) cout << "successful!!!" << endl;
    MPI_Finalize();

    return 0;
}
