#pragma once
#include <map>
#include <algorithm>
#include <list>
#include <vector>
#include <set>
#include <cstdint>
#include <type_traits>
#include "mpi.h"
#include <memory.h>
#include <stdio.h>

//一个重心坐标
typedef struct Barycentric {
	int gvrtx[3];	//三个全局顶点的索引
	short coord[3];	//三个坐标分量
	int newgid;	//一个新的全局顶点的索引
} Barycentric;

typedef struct Baryvrtx {
	int gvrtx[3]; // in ascending order of global vertex ids (starting with 1)
}Barycvrtx;

typedef struct xdFace {
	int lsvrtx[3];
	Barycentric barycv[3];
	short outw;
	short patbound;
	short geoboundary;
}xdFace;

struct FaceRecord {
	std::int32_t lsvrtx[3];
	std::int32_t bary_gvrtx[3][3];
	std::int16_t bary_coord[3][3];
	std::int16_t outw;
	std::int16_t patbound;
	std::int32_t geoboundary;
};

struct TetRecord {
	std::int32_t vids[4];
	std::int32_t domidx;
};

static_assert(std::is_trivially_copyable<FaceRecord>::value, "FaceRecord must be trivially copyable");
static_assert(std::is_trivially_copyable<TetRecord>::value, "TetRecord must be trivially copyable");

typedef struct xdMeshFaceInfo {
	int procids[2];
	int svrtx[2][3];
	int domainidx[2];
	short outw;
} xdMeshFaceInfo;

class fid_xdMeshFaceInfo {
public:
	int fid;
	xdMeshFaceInfo mfi;

	static void build_MPIType();
	static MPI_Datatype MPI_type;
};

typedef struct FacePoints {
	int p[3];
}FacePoints;

struct FacePointCompare {
	bool operator()(const FacePoints & first, const FacePoints & second) const {
		return (first.p[0] < second.p[0]) ||
			(first.p[0] == second.p[0] && first.p[1] < second.p[1]) ||
			(first.p[0] == second.p[0] && first.p[1] == second.p[1] && first.p[2] < second.p[2]);
	}
};

struct CompBarycentric {
	bool operator()(const Barycentric & first, const Barycentric & second) const {
		for (int i = 0; i < 3; i++) {
			if (first.gvrtx[i] < second.gvrtx[i]) {
				return(true);
			}
			else if (first.gvrtx[i] > second.gvrtx[i]) {
				return(false);
			}
		}
		for (int i = 0; i < 3; i++) {
			if (first.coord[i] < second.coord[i]) {
				return(true);
			}
			else if (first.coord[i] > second.coord[i]) {
				return(false);
			}
		}
		return(false);
	}
};

struct CompBarycvrtx {
	bool operator()(const Barycvrtx & first, const Barycvrtx & second) const {
		for (int i = 0; i < 3; i++) {
			if (first.gvrtx[i] < second.gvrtx[i]) {
				return(true);
			}
			else if (first.gvrtx[i] > second.gvrtx[i]) {
				return(false);
			}
		}
		return(false);
	}
};

typedef struct xdVertex {
	double xyz[3];
}xdVertex;

typedef struct IntPair {
	int x;
	int y;
}IntPair;

typedef struct VEindex {
	int gid;
	int Isin;
}VEindex;

typedef struct xdVElement {
	int Pindex[4];
	xdVertex Vertexs[4];
	int gid;
	int domidx;
}xdVElement;

typedef struct xdVElementTet10 {
	int Pindex[10];
	xdVertex Vertexs[10];
	int gid;
	int domidx;
}xdVElementTet10;

struct IntPairCompare {
	bool operator()(const IntPair & first, const IntPair & second) const {
		return first.x < second.x || (first.x == second.x && first.y < second.y);
	}
};

typedef std::map<FacePoints, int, FacePointCompare> surfMap_t;
typedef std::vector<Barycentric> BarycVector;
typedef std::vector<xdVElement>VEVector;
typedef std::vector<xdVElementTet10> VEVectorTet10;

typedef struct SortedKey {
	int id;
	double xyz[3];
}SortedKey;

typedef SortedKey *pSortedKey;


/*memory allocation routines *///�ڴ��������
#define MYMALLOC(v,type,size){ v = (type) malloc(size) ;  \
								if (v == NULL) {\
										printf("Memory alloc error: %1lu\n", size); \
										abort();\
									}\
								}

#define MYREALLOC(v,type,size) { v = (type) realloc(v,size) ;\
								if (v == NULL) {\
										printf("Memory alloc error:%1lu\n", size); \
										abort();\
									}\
								}

#define MYCALLOC(v,type, n,size){ v = (type) calloc(n, size) ;\
								if (v == NULL) {\
										abort();\
									}\
								}





