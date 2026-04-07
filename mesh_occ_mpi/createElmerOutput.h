
#include <vector>
#include <map>
#include <list>
#include <set>
#include <fstream>
#include <sstream>
#include <string>

namespace nglib {
#include <nglib.h>
}
class Index3
{
    public:
        int x[3];

        Index3()
        {
            x[0] = 0;
            x[1] = 0;
            x[2] = 0;
        }

        Index3(int _x, int _y, int _z)
        {
            x[0] = _x;
            x[1] = _y;
            x[2] = _z;
        }

        void swapel(int a, int b)
        {
            int tmp;
            tmp = x[a];
            x[a] = x[b];
            x[b] = tmp;
        }

        void Sort()
        {
            if(x[1] < x[0]) swapel(1,0);
            if(x[2] < x[1]) swapel(2,1);
            if(x[1] < x[0]) swapel(1,0);
        }
};

//判断两个不同Index3内的三个参数是否有序
bool fncomp(Index3 in1, Index3 in2)
{
    if(in1.x[0] < in2.x[0])
        return true;
    else if(in1.x[0] > in2.x[0])
        return false;

    if(in1.x[1] < in2.x[1])
        return true;
    else if(in1.x[1] > in2.x[1])
        return false;

    if(in1.x[2] < in2.x[2])
        return true;
    else if(in1.x[2] > in2.x[2])
        return false;

    return false;
}

#if 0
void createElmerOutput(void *submesh,int *volumegid,int *pointgid,std::map<int, std::list<int>> &adjbarycs,
                       int numprocs,int mypid,string &output_path){
    nglib::Ng_Mesh* mesh = (nglib::Ng_Mesh *)submesh;

    //创建输出文件
    char * boundaryfile1 = new char[512];
    char * elementfile1 = new char[512];
    char * headerfile1 = new char[512];
    char * nodefile1 = new char[512];
    char * sharedfile1 = new char[512];
    char * path1 = new char[512];

    sprintf(path1,"partitioning.%d",numprocs);
    sprintf(boundaryfile1, "partitioning.%d/part.%d.boundary", numprocs, mypid+1);
    sprintf(elementfile1, "partitioning.%d/part.%d.elements", numprocs, mypid+1);
    sprintf(headerfile1, "partitioning.%d/part.%d.header", numprocs, mypid+1);
    sprintf(nodefile1, "partitioning.%d/part.%d.nodes", numprocs, mypid+1);
    sprintf(sharedfile1, "partitioning.%d/part.%d.shared", numprocs, mypid+1);

    string path = output_path + string(path1);
    string boundaryfile = output_path + string(boundaryfile1);
    string elementfile = output_path + string(elementfile1);
    string headerfile = output_path + string(headerfile1);
    string nodefile = output_path + string(nodefile1);
    string sharedfile = output_path + string(sharedfile1);

    mkdir(path.c_str(),0777);

    int ne = nglib::Ng_GetNE(mesh); //体网格的数量
    int nse = nglib::Ng_GetNSE(mesh); //面网格的数量
    int np = nglib::Ng_GetNP(mesh); //点的数量

    //输出elements文件
    ofstream outelements(elementfile.c_str());
    int tet[4];
    for(int i=0;i < ne;i++){
        nglib::Ng_GetVolumeElement (mesh, i+1, tet);
        outelements << volumegid[i+1] << " 1 504 " << pointgid[tet[0]] << " " << pointgid[tet[1]] << " " << pointgid[tet[2]] << " " << pointgid[tet[3]] << endl;
    }
    outelements.close();

    //输出nodes文件
    ofstream outnodes(nodefile.c_str());
    double point[3];
    for(int i=0; i<np;i++){
        nglib::Ng_GetPoint (mesh, i+1, point);
        outnodes << pointgid[i+1] << " -1 " << point[0] << " " << point[1] << " " << point[2] << endl;
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
        sharednode += std::to_string(mypid+1); //elmerID = 核心ID + 1
        sharednode += " ";
        int m = 0;
        for(auto itt = proceid.begin(); itt != proceid.end() ; itt++){
            sharednode += std::to_string(((*itt)+1));     //elmerID = 核心ID + 1
            if( m != (sizeid-1) ){
                sharednode += " ";
            }
            m++;
        }
        outshareds << pointgid[locid] << " " << sharednode << endl;
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
                    i3.x[l] = pointgid[tet[k-1]];
                    l++;
                }
            }
            i3.Sort();
            face2vol.insert(pair<Index3,int>(i3,volumegid[i]));
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
        if(nglib::ispatbound(mesh,j+1)){
            continue;
        }
        i3.x[0] = pointgid[surfpointss[0]];
        i3.x[1] = pointgid[surfpointss[1]];
        i3.x[2] = pointgid[surfpointss[2]];
        i3.Sort();
        myit = face2vol.find(i3);
        if(myit!= face2vol.end()){
            number++;
            outboundarys << number << " " << geoid << " " << myit->second << " " << "0" << " 303 " << pointgid[surfpointss[0]]  << " " << pointgid[surfpointss[1]] << " " << pointgid[surfpointss[2]] <<endl;
        }
    }
    outboundarys.close();


/*
    nglib::My_WriteElmerBound(submesh,boundaryfile);

*/

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
}
#endif