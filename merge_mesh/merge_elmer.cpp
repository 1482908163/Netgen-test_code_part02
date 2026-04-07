#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#include "TopTools_IndexedMapOfShape.hxx"
#include "TopoDS.hxx"
#include "TopoDS_Face.hxx"
#include "TopoDS_Shape.hxx"
#include "GProp_GProps.hxx"
#include "BRepGProp.hxx"
#include "time.h"
#include <filesystem>
namespace nglib {
    #include <nglib.h>
}
#include <dirent.h>
#include <climits>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace nglib;

void print_help() {
    cerr << "参数输入帮助" << endl <<
	"-o : 输出文件目录 默认为input_path+/merge_mesh" << endl <<
	"-i : 输入文件目录 默认为/work/home/moussa/HPDS_test/install/output/wholewall3solid_r0_l0_4_max1000.0_min0.0/volfined" << endl <<
	"-h --help : 参数输入帮助" << endl;
}

vector<string> get_file_number(string path) {
    vector<string> ret;
    DIR *dir;
    struct dirent *ptr;
    if(!(dir = opendir(path.c_str()))) {
        return ret;
    } 
    while((ptr=readdir(dir))!=0) {
        if(strcmp(ptr->d_name,".")!=0 && strcmp(ptr->d_name,"..")!=0) {
            if(ptr->d_type == 8) {
                string file_name = ptr->d_name;
                ret.push_back(file_name);
            } else {
                continue;
            }
        };
    }
    closedir(dir);
    return ret;
}


int main(int argc, char **argv) {

    // string Output_Path = "/work/home/moussa/HPDS/install/output/zhengfangti_r0_l4_4_max8.0_min0.0/merge_mesh/";
    string Input_Path = "/work/home/moussa/HPDS_test/install/output/wholewall3solid_r0_l0_4_max1000.0_min0.0/volfined";
    double start,end;
    start = clock();
    string Output_Path = "";
    // if(argc <= 1) {
    //     print_help();
    //     return 0;
    // }
    for(int i = 1; i < argc; i++) {
        if(!strcmp(argv[i],"-o")) {
			if(argv[i+1] != NULL) Output_Path = argv[i+1];
			else {
				print_help();
				return 1;
			}
		} else if(!strcmp(argv[i],"-i")) {
			if(argv[i+1] != NULL) {
                Input_Path = argv[i+1];
                
            }
			else {
				print_help();
				return 1;
			}
		} else if(!strcmp(argv[i],"-h") || !strcmp(argv[i],"--help")) {
            print_help();
            return 1;
        }
    }

    if(Output_Path == "") Output_Path = Input_Path + "/../merge_mesh";

    vector<string> File_Names = get_file_number(Input_Path);
    Ng_Mesh *my_mesh;
    // Ng_Meshing_Parameters mp;
    // Ng_Result ng_res;
    Ng_Init();

    for(int i = 0; i < File_Names.size(); i++) {
        string path = Input_Path + "/" + File_Names[i];
        my_mesh = Ng_LoadMesh(path.c_str());
        My_Delete_Last_SurfaceDescriptor(my_mesh);
        Ng_SaveMesh(my_mesh, path.c_str());
//        cout << " i : " << i << endl;
    }
    string path = Input_Path + "/"+ File_Names[0];
    my_mesh = Ng_LoadMesh(path.c_str());
    for(int i = 1; i < File_Names.size(); i++) {
        path = Input_Path +"/"+ File_Names[i];
        Ng_Mesh *tmp_mesh;
        tmp_mesh = Ng_LoadMesh(path.c_str());
        Ng_MergeMesh(my_mesh, tmp_mesh);
        //cout << " i : " << i << endl;
    }


    //Ng_SaveMesh(my_mesh, "/work/home/moussa/HPDS/install/output/merge_mesh.vol");

    mkdir(Output_Path.c_str(), 0777);
    cout << Output_Path << endl;

    
    Ng_SaveMesh(my_mesh, (Output_Path + "/merge_mesh.vol").c_str());

    Output_Path = Output_Path + "/test_mesh";
    const std::filesystem::path  &outfile = Output_Path;
    nglib::My_WriteElmerFormat(my_mesh, outfile);

    end = clock();

    cout <<  "spend time : " << (end - start)/CLOCKS_PER_SEC << "s"  << endl;

    return 0;
}
