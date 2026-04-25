#include <map>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <math.h>
#include <iostream>
#include <sstream>
#include <sys/resource.h>
#include <unordered_map>
#include "3DNgmesher.h"
#include <string>
#include <time.h>

// #define M_PI 3.1415926535897932

namespace nglib
{
#include <nglib.h>
}

namespace {

constexpr std::size_t kDefaultStreamBatch = 4096;

const int kTetEdges[6][2] = {
	{0, 1},
	{0, 2},
	{0, 3},
	{1, 2},
	{1, 3},
	{2, 3},
};

const int kTetRefTab[8][4] = {
	{0, 4, 5, 6},
	{4, 1, 7, 8},
	{5, 7, 2, 9},
	{6, 8, 9, 3},
	{4, 5, 6, 8},
	{4, 5, 8, 7},
	{5, 6, 8, 9},
	{5, 7, 9, 8},
};

using EdgeKey = std::uint64_t;

struct EdgeReqRecord {
	EdgeKey key;
	int v0;
	int v1;
};

struct EdgeMidRecord {
	EdgeKey key;
	int mid;
};

struct BoundaryTetFaceRecord
{
	int a;
	int b;
	int c;
	int vegid;
};

struct BoundarySurfaceRecord
{
	int key_a;
	int key_b;
	int key_c;
	int geoid;
	int v0;
	int v1;
	int v2;
};

void VecSub(const double a[3], const double b[3], double out[3])
{
	out[0] = a[0] - b[0];
	out[1] = a[1] - b[1];
	out[2] = a[2] - b[2];
}

double Dot3(const double a[3], const double b[3])
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void Cross3(const double a[3], const double b[3], double out[3])
{
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

double TetVolumeFromPoints(const double p0[3],
	const double p1[3],
	const double p2[3],
	const double p3[3])
{
	double a[3];
	double b[3];
	double c[3];
	double cross_bc[3];
	VecSub(p1, p0, a);
	VecSub(p2, p0, b);
	VecSub(p3, p0, c);
	Cross3(b, c, cross_bc);
	return fabs(Dot3(a, cross_bc)) / 6.0;
}

double GetCurrentRssMb();
double GetPeakRssMb();

void PrintStreamStageMem(const char *tag, const std::string &output_path)
{
	const double local_current = GetCurrentRssMb();
	const double local_peak = GetPeakRssMb();
	int id = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	std::ostringstream oss;
	oss << "[MEM] " << tag << " : current=" << local_current
	    << " MB, peak=" << local_peak << " MB";
	AppendRankTestoutLine(output_path, id, oss.str());
}

std::size_t BoundaryFaceShardIndex(const FaceKey &key, std::size_t num_shards)
{
	if (num_shards == 0)
	{
		return 0;
	}
	return FaceKeyHash{}(key) % num_shards;
}

MPI_Datatype CreateMpiGhostVeType()
{
	StreamGhostVE dummy{};
	int blocklens[4] = {1, 1, 4, 12};
	MPI_Aint addrs[4];
	MPI_Datatype mpitypes[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
	MPI_Datatype dtype;

	MPI_Get_address(&dummy.gid, &addrs[0]);
	MPI_Get_address(&dummy.domidx, &addrs[1]);
	MPI_Get_address(&dummy.pindex, &addrs[2]);
	MPI_Get_address(&dummy.xyz, &addrs[3]);

	addrs[3] -= addrs[0];
	addrs[2] -= addrs[0];
	addrs[1] -= addrs[0];
	addrs[0] = static_cast<MPI_Aint>(0);

	MPI_Type_create_struct(4, blocklens, addrs, mpitypes, &dtype);
	MPI_Type_commit(&dtype);
	return dtype;
}

static_assert(std::is_trivially_copyable<EdgeReqRecord>::value, "EdgeReqRecord must be trivially copyable");
static_assert(std::is_trivially_copyable<EdgeMidRecord>::value, "EdgeMidRecord must be trivially copyable");
static_assert(std::is_trivially_copyable<BoundaryTetFaceRecord>::value, "BoundaryTetFaceRecord must be trivially copyable");
static_assert(std::is_trivially_copyable<BoundarySurfaceRecord>::value, "BoundarySurfaceRecord must be trivially copyable");

void FailStreamOperation(const std::string &message);
void EnsureParentDirectory(const std::string &filepath);
int GetPointRecordCount(const std::string &path);

std::string g_point_table_stream_path;
std::fstream g_point_table_stream;

void ResetPointTableStreamCache()
{
	if (g_point_table_stream.is_open())
	{
		g_point_table_stream.close();
	}
	g_point_table_stream.clear();
	g_point_table_stream_path.clear();
}

std::fstream &OpenPointTableStream(const std::string &path)
{
	if (!g_point_table_stream.is_open() || g_point_table_stream_path != path)
	{
		ResetPointTableStreamCache();
		EnsureParentDirectory(path);
		g_point_table_stream.open(path, std::ios::binary | std::ios::in | std::ios::out);
		if (!g_point_table_stream)
		{
			g_point_table_stream.clear();
			std::ofstream create(path, std::ios::binary | std::ios::trunc);
			if (!create)
			{
				FailStreamOperation("failed to create point table file: " + path);
			}
			create.close();
			g_point_table_stream.open(path, std::ios::binary | std::ios::in | std::ios::out);
		}
		if (!g_point_table_stream)
		{
			FailStreamOperation("failed to open point table file: " + path);
		}
		g_point_table_stream_path = path;
	}
	return g_point_table_stream;
}

void AppendPointRecord(const std::string &path, const double xyz[3])
{
	PointCoordRecord record{};
	record.xyz[0] = xyz[0];
	record.xyz[1] = xyz[1];
	record.xyz[2] = xyz[2];

	std::fstream &stream = OpenPointTableStream(path);
	stream.clear();
	stream.seekp(0, std::ios::end);
	stream.write(reinterpret_cast<const char *>(&record), static_cast<std::streamsize>(sizeof(PointCoordRecord)));
	stream.flush();
	if (!stream)
	{
		FailStreamOperation("failed to append point record to: " + path);
	}
}

void ReadPointRecord(const std::string &path, int point_id, double xyz[3])
{
	if (point_id < 1)
	{
		FailStreamOperation("invalid point id for point table lookup: " + std::to_string(point_id));
	}
	const std::uintmax_t filesize = std::filesystem::exists(path) ? std::filesystem::file_size(path) : 0;
	const std::uintmax_t offset = static_cast<std::uintmax_t>(point_id - 1) * sizeof(PointCoordRecord);
	if (offset + sizeof(PointCoordRecord) > filesize)
	{
		FailStreamOperation("point id out of range for point table lookup: " + std::to_string(point_id));
	}

	PointCoordRecord record{};
	std::fstream &stream = OpenPointTableStream(path);
	stream.flush();
	stream.clear();
	stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
	stream.read(reinterpret_cast<char *>(&record), static_cast<std::streamsize>(sizeof(PointCoordRecord)));
	if (!stream)
	{
		FailStreamOperation("failed to read point record from: " + path);
	}
	xyz[0] = record.xyz[0];
	xyz[1] = record.xyz[1];
	xyz[2] = record.xyz[2];
}

void AppendPointRecordsBulk(const std::string &path, const std::vector<PointCoordRecord> &records)
{
	if (records.empty())
	{
		return;
	}

	std::fstream &stream = OpenPointTableStream(path);
	stream.clear();
	stream.seekp(0, std::ios::end);
	stream.write(reinterpret_cast<const char *>(records.data()),
		static_cast<std::streamsize>(records.size() * sizeof(PointCoordRecord)));
	stream.flush();
	if (!stream)
	{
		FailStreamOperation("failed to append point records to: " + path);
	}
}

std::vector<PointCoordRecord> ReadPointRecordsByIds(const std::string &path,
	const std::vector<int> &sorted_unique_point_ids)
{
	std::vector<PointCoordRecord> coords(sorted_unique_point_ids.size());
	if (sorted_unique_point_ids.empty())
	{
		return coords;
	}

	const int point_count = GetPointRecordCount(path);
	std::fstream &stream = OpenPointTableStream(path);
	stream.flush();
	for (std::size_t i = 0; i < sorted_unique_point_ids.size(); ++i)
	{
		const int point_id = sorted_unique_point_ids[i];
		if (point_id < 1 || point_id > point_count)
		{
			FailStreamOperation("point id out of range for bulk point table lookup: " + std::to_string(point_id));
		}
		if (i > 0 && sorted_unique_point_ids[i - 1] >= point_id)
		{
			FailStreamOperation("point id cache request must be sorted and unique");
		}
		const std::uintmax_t offset = static_cast<std::uintmax_t>(point_id - 1) * sizeof(PointCoordRecord);
		stream.clear();
		stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
		stream.read(reinterpret_cast<char *>(&coords[i]), static_cast<std::streamsize>(sizeof(PointCoordRecord)));
		if (!stream)
		{
			FailStreamOperation("failed to bulk-read point record from: " + path);
		}
	}
	return coords;
}

const PointCoordRecord &LookupPointCoordInCache(const std::vector<int> &sorted_unique_point_ids,
	const std::vector<PointCoordRecord> &coords,
	int point_id)
{
	const auto it = std::lower_bound(sorted_unique_point_ids.begin(), sorted_unique_point_ids.end(), point_id);
	if (it == sorted_unique_point_ids.end() || *it != point_id)
	{
		FailStreamOperation("missing point id in shard point cache: " + std::to_string(point_id));
	}
	const std::size_t index = static_cast<std::size_t>(it - sorted_unique_point_ids.begin());
	if (index >= coords.size())
	{
		FailStreamOperation("point cache index out of range for point id: " + std::to_string(point_id));
	}
	return coords[index];
}

int GetPointRecordCount(const std::string &path)
{
	if (!std::filesystem::exists(path))
	{
		return 0;
	}
	const std::uintmax_t filesize = std::filesystem::file_size(path);
	if (filesize % sizeof(PointCoordRecord) != 0)
	{
		FailStreamOperation("point table file is truncated: " + path);
	}
	return static_cast<int>(filesize / sizeof(PointCoordRecord));
}

int CountRecordsByFileSize(const std::string &path, std::size_t record_size)
{
	if (!std::filesystem::exists(path))
	{
		FailStreamOperation("stream file does not exist: " + path);
	}
	const std::uintmax_t filesize = std::filesystem::file_size(path);
	if (record_size == 0 || filesize % record_size != 0)
	{
		FailStreamOperation("stream file size is not aligned to record size: " + path);
	}
	return static_cast<int>(filesize / record_size);
}

Barycvrtx ToBarycvrtxKey(const Barycentric &bary)
{
	Barycvrtx key;
	key.gvrtx[0] = bary.gvrtx[0];
	key.gvrtx[1] = bary.gvrtx[1];
	key.gvrtx[2] = bary.gvrtx[2];
	return key;
}

std::string DescribeBarycentricKey(const Barycentric &bary)
{
	std::ostringstream oss;
	oss << "gvrtx=("
	    << bary.gvrtx[0] << ","
	    << bary.gvrtx[1] << ","
	    << bary.gvrtx[2] << ") coord=("
	    << bary.coord[0] << ","
	    << bary.coord[1] << ","
	    << bary.coord[2] << ")";
	return oss.str();
}

std::vector<int> NormalizeSharedAdjRanks(const std::list<int> &adj_list, int self_id)
{
	std::vector<int> adj(adj_list.begin(), adj_list.end());
	std::sort(adj.begin(), adj.end());
	adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
	adj.erase(std::remove(adj.begin(), adj.end(), self_id), adj.end());
	return adj;
}

FaceRecord ReadFaceRecordByIndex(const std::string &path, int sid)
{
	if (sid < 1)
	{
		FailStreamOperation("invalid surface element id for stream file lookup: " + std::to_string(sid));
	}
	std::ifstream input(path, std::ios::binary);
	if (!input)
	{
		FailStreamOperation("failed to open surface stream file: " + path);
	}
	const std::streamoff offset = static_cast<std::streamoff>(sid - 1) * static_cast<std::streamoff>(sizeof(FaceRecord));
	input.seekg(offset, std::ios::beg);
	FaceRecord record{};
	input.read(reinterpret_cast<char *>(&record), static_cast<std::streamsize>(sizeof(FaceRecord)));
	if (!input)
	{
		FailStreamOperation("failed to read surface stream record: " + path);
	}
	return record;
}

TetRecord ReadTetRecordByIndex(const std::string &path, int eid)
{
	if (eid < 1)
	{
		FailStreamOperation("invalid volume element id for stream file lookup: " + std::to_string(eid));
	}
	std::ifstream input(path, std::ios::binary);
	if (!input)
	{
		FailStreamOperation("failed to open tet stream file: " + path);
	}
	const std::streamoff offset = static_cast<std::streamoff>(eid - 1) * static_cast<std::streamoff>(sizeof(TetRecord));
	input.seekg(offset, std::ios::beg);
	TetRecord record{};
	input.read(reinterpret_cast<char *>(&record), static_cast<std::streamsize>(sizeof(TetRecord)));
	if (!input)
	{
		FailStreamOperation("failed to read tet stream record: " + path);
	}
	return record;
}

void FailStreamOperation(const std::string &message)
{
	std::cerr << message << std::endl;
	std::abort();
}

void EnsureParentDirectory(const std::string &filepath)
{
	const std::filesystem::path path(filepath);
	if (path.has_parent_path())
	{
		std::filesystem::create_directories(path.parent_path());
	}
}

IntPair MakeOrderedEdge(int a, int b)
{
	IntPair pr;
	if (a < b)
	{
		pr.x = a;
		pr.y = b;
	}
	else
	{
		pr.x = b;
		pr.y = a;
	}
	return pr;
}

void RegisterBarycentricVertex(std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	const Barycentric &bary,
	int vertex_id)
{
	// 任何新增点都必须同时维护正向/反向两张表：
	// baryc2locvrtxmap 用于“同一个重心点只建一次”，
	// locvrtx2barycmap 用于后续从 submesh 反查重心坐标并重建 stream。
	baryc2locvrtxmap[bary] = vertex_id;
	locvrtx2barycmap[vertex_id] = bary;
}

FaceRecord ToFaceRecord(const xdFace &face)
{
	// 将内存中的 xdFace 压成定长二进制记录。
	// 这里不写 Barycentric::newgid，只保留恢复拓扑和 barycentric 所需的最小字段，
	// 这样 face stream 的格式稳定，不依赖复杂对象内存布局。
	FaceRecord record{};
	for (int i = 0; i < 3; ++i)
	{
		record.lsvrtx[i] = face.lsvrtx[i];
		for (int j = 0; j < 3; ++j)
		{
			record.bary_gvrtx[i][j] = face.barycv[i].gvrtx[j];
			record.bary_coord[i][j] = face.barycv[i].coord[j];
		}
	}
	record.outw = face.outw;
	record.patbound = face.patbound;
	record.geoboundary = face.geoboundary;
	return record;
}

xdFace FromFaceRecord(const FaceRecord &record)
{
	// 从定长记录恢复出工作态 xdFace。
	// newgid 在 stream 中没有意义，这里统一置 0，避免把旧内存垃圾带回流程。
	xdFace face{};
	for (int i = 0; i < 3; ++i)
	{
		face.lsvrtx[i] = record.lsvrtx[i];
		for (int j = 0; j < 3; ++j)
		{
			face.barycv[i].gvrtx[j] = record.bary_gvrtx[i][j];
			face.barycv[i].coord[j] = record.bary_coord[i][j];
		}
		face.barycv[i].newgid = 0;
	}
	face.outw = record.outw;
	face.patbound = record.patbound;
	face.geoboundary = record.geoboundary;
	return face;
}

TetRecord ToTetRecord(const int vids[4], int domidx)
{
	// TetRecord 只保留 volume refine 所需的最小信息：
	// 四面体 4 个顶点编号 + 域号。
	TetRecord record{};
	for (int i = 0; i < 4; ++i)
	{
		record.vids[i] = vids[i];
	}
	record.domidx = domidx;
	return record;
}

void FromTetRecord(const TetRecord &record, int vids[4], int &domidx)
{
	// 将定长四面体记录恢复成 refine 逻辑需要的局部数组。
	for (int i = 0; i < 4; ++i)
	{
		vids[i] = record.vids[i];
	}
	domidx = record.domidx;
}

void FlushFaceBuffer(std::ofstream &output, std::vector<FaceRecord> &buffer)
{
	// 统一的批量刷盘函数，避免业务函数里散落重复的 write/error 处理。
	if (buffer.empty())
	{
		return;
	}
	output.write(reinterpret_cast<const char *>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(FaceRecord)));
	if (!output)
	{
		FailStreamOperation("failed to write face stream file");
	}
	buffer.clear();
}

void FlushTetBuffer(std::ofstream &output, std::vector<TetRecord> &buffer)
{
	// volume stream 对应的批量写盘函数。
	if (buffer.empty())
	{
		return;
	}
	output.write(reinterpret_cast<const char *>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(TetRecord)));
	if (!output)
	{
		FailStreamOperation("failed to write tet stream file");
	}
	buffer.clear();
}

std::size_t EffectiveBatch(std::size_t batch_size)
{
	// 用户没显式给 batch 时，统一回落到默认批大小。
	return batch_size == 0 ? kDefaultStreamBatch : batch_size;
}

double GetCurrentRssMb()
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

double GetPeakRssMb()
{
	struct rusage usage {};
	if (getrusage(RUSAGE_SELF, &usage) != 0)
	{
		return 0.0;
	}
	return usage.ru_maxrss / 1024.0;
}

bool IsPowerOfTwo(std::size_t value)
{
	return value != 0 && (value & (value - 1)) == 0;
}

void PrintInnerMemLog(const char *phase,
	int round,
	const std::string &output_path,
	void *submesh,
	const std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	const std::map<int, Barycentric> &locvrtx2barycmap,
	const std::map<IntPair, int, IntPairCompare> &edgemap,
	const std::size_t *boundary_edges_size = nullptr,
	const std::size_t *output_buffer_size = nullptr,
	const std::size_t *total_unique_internal_edges = nullptr)
{
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	nglib::Ng_Mesh *mesh = (nglib::Ng_Mesh *)submesh;

	std::ostringstream oss;
	oss << "[MEM-INNER] rank=" << rank
		<< " round=" << round
		<< " phase=" << phase
		<< " current_rss=" << GetCurrentRssMb() << " MB"
		<< " peak_rss=" << GetPeakRssMb() << " MB"
		<< " np=" << nglib::Ng_GetNP(mesh)
		<< " baryc2locvrtxmap.size=" << baryc2locvrtxmap.size()
		<< " locvrtx2barycmap.size=" << locvrtx2barycmap.size()
		<< " edgemap.size=" << edgemap.size();
	if (boundary_edges_size != nullptr)
	{
		oss << " boundary_edges.size=" << *boundary_edges_size;
	}
	if (output_buffer_size != nullptr)
	{
		oss << " output_buffer.size=" << *output_buffer_size;
	}
	if (total_unique_internal_edges != nullptr)
	{
		oss << " total_unique_internal_edges=" << *total_unique_internal_edges;
	}
	AppendRankTestoutLine(output_path, rank, oss.str());
}

std::size_t ReadFaceBatch(std::ifstream &input, std::vector<FaceRecord> &buffer, std::size_t batch_size)
{
	// 按固定记录长度读取一个批次；如果文件长度不是记录大小的整数倍，直接判为损坏。
	buffer.resize(EffectiveBatch(batch_size));
	input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(FaceRecord)));
	const std::streamsize bytes_read = input.gcount();
	if (bytes_read < 0 || bytes_read % static_cast<std::streamsize>(sizeof(FaceRecord)) != 0)
	{
		FailStreamOperation("face stream file is truncated");
	}
	const std::size_t count = static_cast<std::size_t>(bytes_read / static_cast<std::streamsize>(sizeof(FaceRecord)));
	buffer.resize(count);
	return count;
}

std::size_t ReadTetBatch(std::ifstream &input, std::vector<TetRecord> &buffer, std::size_t batch_size)
{
	// 与 face stream 相同，按定长 TetRecord 批量读取。
	buffer.resize(EffectiveBatch(batch_size));
	input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(TetRecord)));
	const std::streamsize bytes_read = input.gcount();
	if (bytes_read < 0 || bytes_read % static_cast<std::streamsize>(sizeof(TetRecord)) != 0)
	{
		FailStreamOperation("tet stream file is truncated");
	}
	const std::size_t count = static_cast<std::size_t>(bytes_read / static_cast<std::streamsize>(sizeof(TetRecord)));
	buffer.resize(count);
	return count;
}

EdgeKey PackEdgeKey(int a, int b)
{
	const IntPair ordered = MakeOrderedEdge(a, b);
	return (static_cast<EdgeKey>(static_cast<std::uint32_t>(ordered.x)) << 32) |
		static_cast<std::uint32_t>(ordered.y);
}

std::size_t EdgeShardIndex(EdgeKey key, std::size_t num_shards)
{
	return num_shards == 0 ? 0 : static_cast<std::size_t>(key % num_shards);
}

void WriteEdgeReq(std::ofstream &output, std::vector<EdgeReqRecord> &buffer)
{
	if (buffer.empty())
	{
		return;
	}
	output.write(reinterpret_cast<const char *>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(EdgeReqRecord)));
	if (!output)
	{
		FailStreamOperation("failed to write edge request shard file");
	}
	buffer.clear();
}

std::size_t ReadEdgeReqBatch(std::ifstream &input, std::vector<EdgeReqRecord> &buffer, std::size_t batch_records)
{
	buffer.resize(EffectiveBatch(batch_records));
	input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(EdgeReqRecord)));
	const std::streamsize bytes_read = input.gcount();
	if (bytes_read < 0 || bytes_read % static_cast<std::streamsize>(sizeof(EdgeReqRecord)) != 0)
	{
		FailStreamOperation("edge request shard file is truncated");
	}
	const std::size_t count = static_cast<std::size_t>(bytes_read / static_cast<std::streamsize>(sizeof(EdgeReqRecord)));
	buffer.resize(count);
	return count;
}

void WriteEdgeMidRecords(std::ofstream &output, const std::vector<EdgeMidRecord> &records)
{
	if (records.empty())
	{
		return;
	}
	output.write(reinterpret_cast<const char *>(records.data()), static_cast<std::streamsize>(records.size() * sizeof(EdgeMidRecord)));
	if (!output)
	{
		FailStreamOperation("failed to write edge midpoint shard file");
	}
}

std::size_t ReadEdgeMidRecords(std::ifstream &input, std::vector<EdgeMidRecord> &buffer, std::size_t batch_records)
{
	buffer.resize(EffectiveBatch(batch_records));
	input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(EdgeMidRecord)));
	const std::streamsize bytes_read = input.gcount();
	if (bytes_read < 0 || bytes_read % static_cast<std::streamsize>(sizeof(EdgeMidRecord)) != 0)
	{
		FailStreamOperation("edge midpoint shard file is truncated");
	}
	const std::size_t count = static_cast<std::size_t>(bytes_read / static_cast<std::streamsize>(sizeof(EdgeMidRecord)));
	buffer.resize(count);
	return count;
}

int LookupMidpointInShardVector(const std::vector<EdgeMidRecord> &records, EdgeKey key)
{
	const auto it = std::lower_bound(records.begin(), records.end(), key,
		[](const EdgeMidRecord &record, EdgeKey target) {
			return record.key < target;
		});
	if (it == records.end() || it->key != key)
	{
		FailStreamOperation("missing internal edge midpoint for shard key " + std::to_string(key));
	}
	return it->mid;
}

std::vector<std::string> BuildEdgeShardFilePaths(const std::string &tet_outfile,
	const char *prefix,
	int round,
	std::size_t num_shards)
{
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::vector<std::string> paths(num_shards);
	const std::filesystem::path parent_dir = std::filesystem::path(tet_outfile).parent_path();
	for (std::size_t shard = 0; shard < num_shards; ++shard)
	{
		const std::filesystem::path filepath = parent_dir /
			(std::string(prefix) + "_rank" + std::to_string(rank) + "_round" + std::to_string(round) + "_shard" + std::to_string(shard) + ".bin");
		paths[shard] = filepath.string();
	}
	return paths;
}

std::vector<std::string> BuildInteriorEdgeShardFiles(const std::string &tet_infile,
	const std::string &tet_outfile,
	const std::set<IntPair, IntPairCompare> &boundary_edges,
	std::size_t batch_tets,
	int round,
	std::size_t num_shards)
{
	const std::vector<std::string> shard_files = BuildEdgeShardFilePaths(tet_outfile, "edge_req", round, num_shards);
	std::vector<std::ofstream> outputs(num_shards);
	std::vector<std::vector<EdgeReqRecord>> buffers(num_shards);
	for (std::size_t shard = 0; shard < num_shards; ++shard)
	{
		EnsureParentDirectory(shard_files[shard]);
		outputs[shard].open(shard_files[shard], std::ios::binary | std::ios::trunc);
		if (!outputs[shard])
		{
			FailStreamOperation("failed to open edge request shard file: " + shard_files[shard]);
		}
		buffers[shard].reserve(kDefaultStreamBatch);
	}

	std::ifstream input(tet_infile, std::ios::binary);
	if (!input)
	{
		FailStreamOperation("failed to open tet stream input file for edge sharding: " + tet_infile);
	}

	std::vector<TetRecord> input_buffer;
	const std::size_t edge_batch = EffectiveBatch(batch_tets) * 6;
	while (ReadTetBatch(input, input_buffer, batch_tets) > 0)
	{
		for (const TetRecord &record : input_buffer)
		{
			int vids[4];
			int domidx = 0;
			FromTetRecord(record, vids, domidx);
			(void)domidx;
			for (int edge = 0; edge < 6; ++edge)
			{
				const int va = vids[kTetEdges[edge][0]];
				const int vb = vids[kTetEdges[edge][1]];
				const IntPair ordered_edge = MakeOrderedEdge(va, vb);
				if (boundary_edges.find(ordered_edge) != boundary_edges.end())
				{
					continue;
				}
				const EdgeKey key = PackEdgeKey(ordered_edge.x, ordered_edge.y);
				const std::size_t shard = EdgeShardIndex(key, num_shards);
				buffers[shard].push_back(EdgeReqRecord{key, ordered_edge.x, ordered_edge.y});
				if (buffers[shard].size() >= edge_batch)
				{
					WriteEdgeReq(outputs[shard], buffers[shard]);
				}
			}
		}
	}

	for (std::size_t shard = 0; shard < num_shards; ++shard)
	{
		WriteEdgeReq(outputs[shard], buffers[shard]);
	}
	return shard_files;
}

std::vector<std::vector<EdgeMidRecord>> BuildEdgeMidShardTables(const std::vector<std::string> &edge_req_files,
	const std::vector<std::string> &edge_mid_files,
	const std::string &point_table_path,
	std::size_t batch_tets,
	int &next_point_id,
	std::size_t &total_unique_internal_edges)
{
	std::vector<std::vector<EdgeMidRecord>> shard_mid_tables(edge_req_files.size());
	total_unique_internal_edges = 0;

	for (std::size_t shard = 0; shard < edge_req_files.size(); ++shard)
	{
		std::ifstream input(edge_req_files[shard], std::ios::binary);
		if (!input)
		{
			FailStreamOperation("failed to open edge request shard file: " + edge_req_files[shard]);
		}

		std::vector<EdgeReqRecord> req_buffer;
		std::vector<EdgeReqRecord> all_requests;
		const std::size_t edge_batch = EffectiveBatch(batch_tets) * 6;
		while (ReadEdgeReqBatch(input, req_buffer, edge_batch) > 0)
		{
			all_requests.insert(all_requests.end(), req_buffer.begin(), req_buffer.end());
		}

		std::sort(all_requests.begin(), all_requests.end(),
			[](const EdgeReqRecord &lhs, const EdgeReqRecord &rhs) {
				return lhs.key < rhs.key;
			});

			std::vector<EdgeReqRecord> unique_requests;
			unique_requests.reserve(all_requests.size());
			for (std::size_t i = 0; i < all_requests.size();)
			{
				unique_requests.push_back(all_requests[i]);
				const EdgeKey current_key = all_requests[i].key;
				do
				{
					++i;
				}
				while (i < all_requests.size() && all_requests[i].key == current_key);
			}

			std::vector<int> endpoint_ids;
			endpoint_ids.reserve(unique_requests.size() * 2);
			for (const EdgeReqRecord &request : unique_requests)
			{
				endpoint_ids.push_back(request.v0);
				endpoint_ids.push_back(request.v1);
			}
			std::sort(endpoint_ids.begin(), endpoint_ids.end());
			endpoint_ids.erase(std::unique(endpoint_ids.begin(), endpoint_ids.end()), endpoint_ids.end());

			const std::vector<PointCoordRecord> endpoint_coords = ReadPointRecordsByIds(point_table_path, endpoint_ids);

			std::vector<EdgeMidRecord> mid_records;
			mid_records.reserve(unique_requests.size());
			std::vector<PointCoordRecord> point_append_buffer;
			point_append_buffer.reserve(unique_requests.size());
			for (const EdgeReqRecord &request : unique_requests)
			{
				const PointCoordRecord &coord0 = LookupPointCoordInCache(endpoint_ids, endpoint_coords, request.v0);
				const PointCoordRecord &coord1 = LookupPointCoordInCache(endpoint_ids, endpoint_coords, request.v1);
				PointCoordRecord midpoint{};
				midpoint.xyz[0] = 0.5 * (coord0.xyz[0] + coord1.xyz[0]);
				midpoint.xyz[1] = 0.5 * (coord0.xyz[1] + coord1.xyz[1]);
				midpoint.xyz[2] = 0.5 * (coord0.xyz[2] + coord1.xyz[2]);

				const int index = next_point_id++;
				mid_records.push_back(EdgeMidRecord{request.key, index});
				point_append_buffer.push_back(midpoint);
				++total_unique_internal_edges;
			}
			AppendPointRecordsBulk(point_table_path, point_append_buffer);

		std::ofstream output(edge_mid_files[shard], std::ios::binary | std::ios::trunc);
		if (!output)
		{
			FailStreamOperation("failed to open edge midpoint shard file: " + edge_mid_files[shard]);
		}
		WriteEdgeMidRecords(output, mid_records);
		shard_mid_tables[shard] = std::move(mid_records);
	}

	return shard_mid_tables;
}

bool TryGetVertexBarycentric(int vertex_id, const std::map<int, Barycentric> &locvrtx2barycmap, Barycentric &bary)
{
	// 反向查找：从 submesh 本地点号拿回 barycentric。
	// 这是完整体 stream 方案成立的关键，因为 surface/tet dump 都需要从当前 mesh 反查顶点来源。
	const auto it = locvrtx2barycmap.find(vertex_id);
	if (it == locvrtx2barycmap.end())
	{
		return false;
	}
	bary = it->second;
	return true;
}

int GetOrCreateEdgeMidpoint(nglib::Ng_Mesh *submesh,
	int v0,
	int v1,
	const Barycentric *bary0,
	const Barycentric *bary1,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	std::map<IntPair, int, IntPairCompare> &edgemap,
	Barycentric *midpoint_bary,
	const std::string &point_table_path,
	int &next_point_id,
	bool materialize_to_submesh)
{
	// 统一的“边中点获取/创建”入口，surface refine 和 volume refine 都走这里。
	//
	// 规则：
	// 1. 如果这条边对应的 barycentric 中点已存在，以 baryc2locvrtxmap 为准直接复用；
	// 2. 否则再看当前轮 edgemap 是否已有缓存；
	// 3. 两者都没有才真正创建新的中点。
	//
	// 这样可以同时保证：
	// - shared boundary / interface 上中点不重复；
	// - 同一轮 refine 内部重复边不反复建点；
	// - 所有新点都同步维护双向 barycentric 映射。
	const IntPair pr = MakeOrderedEdge(v0, v1);
	const bool has_bary = bary0 != nullptr && bary1 != nullptr;
	const int point_limit = materialize_to_submesh ? nglib::Ng_GetNP(submesh) : (next_point_id - 1);
	if (v0 < 1 || v0 > point_limit || v1 < 1 || v1 > point_limit)
	{
		std::fprintf(stderr, "Invalid local edge vertex id: v0=%d v1=%d point_limit=%d mesh=%p materialize=%d\n",
			v0, v1, point_limit, (void *)submesh, materialize_to_submesh ? 1 : 0);
		if (has_bary)
		{
			std::fprintf(stderr,
				"  bary0=(%d,%d,%d | %d,%d,%d) bary1=(%d,%d,%d | %d,%d,%d)\n",
				bary0->gvrtx[0], bary0->gvrtx[1], bary0->gvrtx[2],
				bary0->coord[0], bary0->coord[1], bary0->coord[2],
				bary1->gvrtx[0], bary1->gvrtx[1], bary1->gvrtx[2],
				bary1->coord[0], bary1->coord[1], bary1->coord[2]);
		}
		std::abort();
	}
	Barycentric midbary{};
	if (has_bary)
	{
		BarycMidPoint(*bary0, *bary1, midbary);
		if (midpoint_bary != nullptr)
		{
			*midpoint_bary = midbary;
		}
		const auto bi = baryc2locvrtxmap.find(midbary);
		if (bi != baryc2locvrtxmap.end())
		{
			edgemap[pr] = bi->second;
			return bi->second;
		}
	}

	const auto ei = edgemap.find(pr);
	if (ei != edgemap.end())
	{
		return ei->second;
	}

	double xyz0[3];
	double xyz1[3];
	double midpoint[3];
	if (materialize_to_submesh)
	{
		nglib::Ng_GetPoint(submesh, v0, xyz0);
		nglib::Ng_GetPoint(submesh, v1, xyz1);
	}
	else
	{
		ReadPointRecord(point_table_path, v0, xyz0);
		ReadPointRecord(point_table_path, v1, xyz1);
	}
	midpoint[0] = 0.5 * (xyz0[0] + xyz1[0]);
	midpoint[1] = 0.5 * (xyz0[1] + xyz1[1]);
	midpoint[2] = 0.5 * (xyz0[2] + xyz1[2]);

	int index = -1;
	if (materialize_to_submesh)
	{
		nglib::Ng_AddPoint(submesh, midpoint, index);
	}
	else
	{
		index = next_point_id++;
		AppendPointRecord(point_table_path, midpoint);
	}
	edgemap[pr] = index;

	if (has_bary)
	{
		RegisterBarycentricVertex(baryc2locvrtxmap, locvrtx2barycmap, midbary, index);
	}
	return index;
}

int GetOrCreateEdgeMidpoint(nglib::Ng_Mesh *submesh,
	int v0,
	int v1,
	const Barycentric *bary0,
	const Barycentric *bary1,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	std::map<IntPair, int, IntPairCompare> &edgemap,
	Barycentric *midpoint_bary)
{
	int next_point_id = nglib::Ng_GetNP(submesh) + 1;
	return GetOrCreateEdgeMidpoint(submesh,
		v0,
		v1,
		bary0,
		bary1,
		baryc2locvrtxmap,
		locvrtx2barycmap,
		edgemap,
		midpoint_bary,
		std::string(),
		next_point_id,
		true);
	}

	void RefineFaceAndCollectChildren(nglib::Ng_Mesh *submesh,
	const xdFace &face,
	std::vector<xdFace> &children,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	std::map<IntPair, int, IntPairCompare> &edgemap,
	std::set<IntPair, IntPairCompare> *boundary_edges,
	const std::string &point_table_path,
	int &next_point_id,
	bool materialize_to_submesh)
{
	// 把一个三角面细分成 4 个子三角形，并返回子面列表。
	// 这是原 Refine()/Refineforvol() 的公共细分逻辑，stream/non-stream 两条路径共用。
	//
	// boundary_edges 非空时，函数会顺手把原边界三角面的 3 条边记录下来，
	// 供 volume stream 判断“哪些四面体边需要走 barycentric 中点规则”。
	int v[3];
	int u[3];
	Barycentric vbarycv[3];
	Barycentric ubarycv[3];
	const int point_limit = materialize_to_submesh ? nglib::Ng_GetNP(submesh) : (next_point_id - 1);

	for (int k = 0; k < 3; ++k)
	{
		// 先一次性把 face 的 3 个顶点和 barycentric 全部读出来。
		// 这里不能边初始化边读下一点，否则会出现读取未初始化顶点的错误。
		v[k] = face.lsvrtx[k];
		vbarycv[k] = face.barycv[k];
			if (v[k] < 1 || v[k] > point_limit)
			{
				std::fprintf(stderr,
					"Invalid face vertex id before refine: v[%d]=%d point_limit=%d face=(%d,%d,%d) geoboundary=%d materialize=%d\n",
					k, v[k], point_limit, face.lsvrtx[0], face.lsvrtx[1], face.lsvrtx[2], face.geoboundary, materialize_to_submesh ? 1 : 0);
			std::abort();
		}
	}

	for (int k = 0; k < 3; ++k)
	{
		// 对三条边逐条生成/复用中点。
		if (boundary_edges != nullptr)
		{
			boundary_edges->insert(MakeOrderedEdge(v[k], v[(k + 1) % 3]));
		}
		u[k] = GetOrCreateEdgeMidpoint(submesh,
			v[k],
			v[(k + 1) % 3],
			&vbarycv[k],
			&vbarycv[(k + 1) % 3],
			baryc2locvrtxmap,
			locvrtx2barycmap,
			edgemap,
			&ubarycv[k],
			point_table_path,
			next_point_id,
			materialize_to_submesh);
	}

	children.clear();
	children.resize(4);
	for (int k = 0; k < 3; ++k)
	{
		xdFace child{};
		child.lsvrtx[0] = v[k];
		child.barycv[0] = vbarycv[k];
		child.lsvrtx[1] = u[k];
		child.barycv[1] = ubarycv[k];
		child.lsvrtx[2] = u[(k + 2) % 3];
		child.barycv[2] = ubarycv[(k + 2) % 3];
		child.outw = face.outw;
		child.patbound = face.patbound;
		child.geoboundary = face.geoboundary;
		children[k] = child;
	}

	xdFace center{};
	center.lsvrtx[0] = u[0];
	center.barycv[0] = ubarycv[0];
	center.lsvrtx[1] = u[1];
	center.barycv[1] = ubarycv[1];
	center.lsvrtx[2] = u[2];
	center.barycv[2] = ubarycv[2];
	center.outw = face.outw;
	center.patbound = face.patbound;
	center.geoboundary = face.geoboundary;
	children[3] = center;
}

void RefineFaceAndCollectChildren(nglib::Ng_Mesh *submesh,
	const xdFace &face,
	std::vector<xdFace> &children,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	std::map<IntPair, int, IntPairCompare> &edgemap,
	std::set<IntPair, IntPairCompare> *boundary_edges)
{
	int next_point_id = nglib::Ng_GetNP(submesh) + 1;
	RefineFaceAndCollectChildren(submesh,
		face,
		children,
		baryc2locvrtxmap,
		locvrtx2barycmap,
		edgemap,
		boundary_edges,
		std::string(),
		next_point_id,
		true);
	}

	void StreamRefineFacesToFile(void *submesh,
	const std::string &infile,
	const std::string &outfile,
	std::size_t batch_faces,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	std::map<IntPair, int, IntPairCompare> &edgemap,
	std::set<IntPair, IntPairCompare> *boundary_edges,
	const std::string &point_table_path,
	int &next_point_id,
	bool materialize_to_submesh)
{
	// surface stream 的核心批处理函数：
	// 从 infile 按批读 FaceRecord -> 还原为 xdFace -> 细分为 4 个子三角 ->
	// 再写入 outfile。
	//
	// 整个过程只保留“当前批 + 当前 face 的 4 个子面”，
	// 不再让 newfaces 在内存中按 4^L 膨胀。
	std::ifstream input(infile, std::ios::binary);
	if (!input)
	{
		FailStreamOperation("failed to open face stream input file: " + infile);
	}
	EnsureParentDirectory(outfile);
	std::ofstream output(outfile, std::ios::binary | std::ios::trunc);
	if (!output)
	{
		FailStreamOperation("failed to open face stream output file: " + outfile);
	}

	std::vector<FaceRecord> input_buffer;
	std::vector<FaceRecord> output_buffer;
	output_buffer.reserve(EffectiveBatch(batch_faces) * 4);
	std::vector<xdFace> children;
	children.reserve(4);

	while (ReadFaceBatch(input, input_buffer, batch_faces) > 0)
	{
		for (const FaceRecord &record : input_buffer)
		{
			const xdFace face = FromFaceRecord(record);
				RefineFaceAndCollectChildren((nglib::Ng_Mesh *)submesh,
					face,
					children,
					baryc2locvrtxmap,
					locvrtx2barycmap,
					edgemap,
					boundary_edges,
					point_table_path,
					next_point_id,
					materialize_to_submesh);
			for (const xdFace &child : children)
			{
				output_buffer.push_back(ToFaceRecord(child));
			}
			if (output_buffer.size() >= EffectiveBatch(batch_faces) * 4)
			{
				FlushFaceBuffer(output, output_buffer);
			}
		}
	}

	FlushFaceBuffer(output, output_buffer);
}

void StreamRefineFacesToFile(void *submesh,
	const std::string &infile,
	const std::string &outfile,
	std::size_t batch_faces,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	std::map<IntPair, int, IntPairCompare> &edgemap,
	std::set<IntPair, IntPairCompare> *boundary_edges)
{
	int next_point_id = nglib::Ng_GetNP((nglib::Ng_Mesh *)submesh) + 1;
	StreamRefineFacesToFile(submesh,
		infile,
		outfile,
		batch_faces,
		baryc2locvrtxmap,
		locvrtx2barycmap,
		edgemap,
		boundary_edges,
		std::string(),
		next_point_id,
		true);
}

} // namespace

void AppendRankTestoutLine(const std::string &output_path,
	int rank,
	const std::string &line)
{
	const std::filesystem::path dir = std::filesystem::path(output_path) / "testout";
	std::filesystem::create_directories(dir);
	const std::filesystem::path path = dir / ("testout_rank" + std::to_string(rank) + ".txt");
	std::ofstream out(path, std::ios::app);
	out << line << '\n';
	out.flush();
}

void DumpInitialPointsToPointTable(void *submesh, const std::string &path)
{
	ResetPointTableStreamCache();
	EnsureParentDirectory(path);
	std::ofstream output(path, std::ios::binary | std::ios::trunc);
	if (!output)
	{
		FailStreamOperation("failed to open point table file for initial dump: " + path);
	}

	nglib::Ng_Mesh *mesh = (nglib::Ng_Mesh *)submesh;
	const int np = nglib::Ng_GetNP(mesh);
	for (int point_id = 1; point_id <= np; ++point_id)
	{
		PointCoordRecord record{};
		nglib::Ng_GetPoint(mesh, point_id, record.xyz);
		output.write(reinterpret_cast<const char *>(&record), static_cast<std::streamsize>(sizeof(PointCoordRecord)));
		if (!output)
		{
			FailStreamOperation("failed while dumping initial points to point table: " + path);
		}
	}
	output.close();
	ResetPointTableStreamCache();
}

void ReplayNewPointsFromPointTableToMesh(void *submesh, const std::string &point_table_path, int start_point_id)
{
	const int total_points = GetPointRecordCount(point_table_path);
	nglib::Ng_Mesh *mesh = (nglib::Ng_Mesh *)submesh;
	for (int point_id = start_point_id; point_id <= total_points; ++point_id)
	{
		double xyz[3];
		int index = -1;
		ReadPointRecord(point_table_path, point_id, xyz);
		nglib::Ng_AddPoint(mesh, xyz, index);
		if (index != point_id)
		{
			FailStreamOperation("point renumbering changed while replaying point table");
		}
	}
}

bool InitStreamMeshView(StreamMeshView &view,
	const std::string &point_table_path,
	const std::string &surface_file_path,
	const std::string &tet_file_path)
{
	view.point_table_path = point_table_path;
	view.surface_file_path = surface_file_path;
	view.tet_file_path = tet_file_path;
	view.np = CountRecordsByFileSize(point_table_path, sizeof(PointCoordRecord));
	view.nse = CountRecordsByFileSize(surface_file_path, sizeof(FaceRecord));
	view.ne = CountRecordsByFileSize(tet_file_path, sizeof(TetRecord));
	return true;
}

int StreamMesh_GetNP(const StreamMeshView &view)
{
	return view.np;
}

int StreamMesh_GetNSE(const StreamMeshView &view)
{
	return view.nse;
}

int StreamMesh_GetNE(const StreamMeshView &view)
{
	return view.ne;
}

void StreamMesh_GetPoint(const StreamMeshView &view, int pid, double xyz[3])
{
	if (pid < 1 || pid > view.np)
	{
		FailStreamOperation("stream mesh point id out of range: " + std::to_string(pid));
	}
	ReadPointRecord(view.point_table_path, pid, xyz);
}

void StreamMesh_GetSurfaceElement(const StreamMeshView &view, int sid, int tri[3], int &surfidx)
{
	if (sid < 1 || sid > view.nse)
	{
		FailStreamOperation("stream mesh surface element id out of range: " + std::to_string(sid));
	}
	const xdFace face = FromFaceRecord(ReadFaceRecordByIndex(view.surface_file_path, sid));
	for (int i = 0; i < 3; ++i)
	{
		tri[i] = face.lsvrtx[i];
	}
	surfidx = face.geoboundary;
}

void StreamMesh_GetVolumeElement(const StreamMeshView &view, int eid, int tet[4], int &domidx)
{
	if (eid < 1 || eid > view.ne)
	{
		FailStreamOperation("stream mesh volume element id out of range: " + std::to_string(eid));
	}
	FromTetRecord(ReadTetRecordByIndex(view.tet_file_path, eid), tet, domidx);
}

int *com_barycoords_from_streams(
	const StreamMeshView &smv,
	MPI_Comm comm,
	const std::map<Barycvrtx, std::list<int>, CompBarycvrtx> &barycvrtx2adjprocsmap,
	const std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, std::list<int>> &adjbarycs,
	int numParts,
	int *VEgid,
	int id)
{
	const int np = StreamMesh_GetNP(smv);
	const int ne = StreamMesh_GetNE(smv);

	adjbarycs.clear();

	int count = 3;
	int blocklens[3];
	MPI_Aint addrs[3];
	MPI_Datatype mpitypes[3];
	MPI_Datatype mpibaryctype;
	int num_s = 0;
	int num_r = 0;
	int *dest = nullptr;
	int *src = nullptr;
	int *s_length = nullptr;
	int *r_length = nullptr;
	Barycentric **s_data = nullptr;
	Barycentric **r_data = nullptr;
	Barycentric brcy{};
	std::map<int, BarycVector *> pidmap;
	std::map<int, int> srcmap;
	int *globoffsets = nullptr;
	int *globoffsetsml = nullptr;
	int *globoffsetsVE = nullptr;
	int *globoffsetsmlVE = nullptr;
	int *newgid = nullptr;
	int newglobalnocounter = 0;
	std::list<int> pids;

	MYCALLOC(newgid, int *, (np + 1), sizeof(int));

	blocklens[0] = 3;
	blocklens[1] = 3;
	blocklens[2] = 1;
	mpitypes[0] = MPI_INT;
	mpitypes[1] = MPI_SHORT;
	mpitypes[2] = MPI_INT;
	MPI_Get_address(&brcy.gvrtx, addrs);
	MPI_Get_address(&brcy.coord, addrs + 1);
	MPI_Get_address(&brcy.newgid, addrs + 2);
	addrs[1] = addrs[1] - addrs[0];
	addrs[2] = addrs[2] - addrs[0];
	addrs[0] = static_cast<MPI_Aint>(0);
	MPI_Type_create_struct(count, blocklens, addrs, mpitypes, &mpibaryctype);
	MPI_Type_commit(&mpibaryctype);

	for (const auto &entry : baryc2locvrtxmap)
	{
		brcy = entry.first;
		const int locid = entry.second;
		if (locid < 1 || locid > np)
		{
			FailStreamOperation("stream barycentric local vertex id out of range: " + std::to_string(locid));
		}

		const Barycvrtx key = ToBarycvrtxKey(brcy);
		const auto ibc = barycvrtx2adjprocsmap.find(key);
		if (ibc == barycvrtx2adjprocsmap.end())
		{
			continue;
		}
		if (ibc->second.empty())
		{
			FailStreamOperation("shared barycentric vertex has empty adjacency list: " + DescribeBarycentricKey(brcy));
		}

		pids = ibc->second;
		adjbarycs[locid] = pids;

		std::vector<int> holders;
		holders.push_back(id);
		for (const int pid : ibc->second)
		{
			holders.push_back(pid);
		}
		std::sort(holders.begin(), holders.end());

		const int indxowner =
			(brcy.gvrtx[0] + brcy.gvrtx[1] + brcy.gvrtx[2] +
			 brcy.coord[0] + brcy.coord[1] + brcy.coord[2]) %
			static_cast<int>(ibc->second.size());
		const int ownerpid = holders[indxowner];
		if (ownerpid == id)
		{
			++newglobalnocounter;
			newgid[locid] = newglobalnocounter;
			brcy.newgid = newglobalnocounter;
			for (const int pid : ibc->second)
			{
				auto ipdt = pidmap.find(pid);
				if (ipdt == pidmap.end())
				{
					pidmap[pid] = new BarycVector();
				}
				pidmap[pid]->push_back(brcy);
			}
		}
		else
		{
			newgid[locid] = -1;
			auto srcit = srcmap.find(ownerpid);
			if (srcit == srcmap.end())
			{
				srcmap[ownerpid] = 1;
			}
			else
			{
				srcit->second = srcit->second + 1;
			}
		}
	}

	for (int locid = 1; locid <= np; ++locid)
	{
		if (newgid[locid] == 0)
		{
			++newglobalnocounter;
			newgid[locid] = newglobalnocounter;
		}
	}

	MYCALLOC(globoffsetsml, int *, (numParts + 1), sizeof(int));
	globoffsets = globoffsetsml + 1;
	globoffsets[-1] = 0;
	MPI_Allgather(&newglobalnocounter, 1, MPI_INT, globoffsets, 1, MPI_INT, comm);
	for (int i = 0; i < numParts; ++i)
	{
		globoffsets[i] += globoffsets[i - 1];
	}
	for (int locid = 1; locid <= np; ++locid)
	{
		if (newgid[locid] != -1)
		{
			newgid[locid] += globoffsets[id - 1];
		}
	}

	MYCALLOC(globoffsetsmlVE, int *, (numParts + 1), sizeof(int));
	globoffsetsVE = globoffsetsmlVE + 1;
	globoffsetsVE[-1] = 0;
	MPI_Allgather(&ne, 1, MPI_INT, globoffsetsVE, 1, MPI_INT, comm);
	for (int i = 0; i < numParts; ++i)
	{
		globoffsetsVE[i] += globoffsetsVE[i - 1];
	}
	for (int locVEid = 1; locVEid <= ne; ++locVEid)
	{
		VEgid[locVEid] = locVEid + globoffsetsVE[id - 1];
	}

	num_s = static_cast<int>(pidmap.size());
	if (num_s > 0)
	{
		MYCALLOC(s_length, int *, num_s, sizeof(int));
		MYCALLOC(dest, int *, num_s, sizeof(int));
		MYCALLOC(s_data, Barycentric **, num_s, sizeof(Barycentric *));
	}
	int send_index = 0;
	for (auto ipdt = pidmap.begin(); ipdt != pidmap.end(); ++ipdt, ++send_index)
	{
		s_length[send_index] = static_cast<int>(ipdt->second->size());
		dest[send_index] = ipdt->first;
		s_data[send_index] = &((*ipdt->second)[0]);
	}

	num_r = static_cast<int>(srcmap.size());
	if (num_r > 0)
	{
		MYCALLOC(r_length, int *, num_r, sizeof(int));
		MYCALLOC(src, int *, num_r, sizeof(int));
		MYCALLOC(r_data, Barycentric **, num_r, sizeof(Barycentric *));
	}
	int recv_index = 0;
	for (auto srcit = srcmap.begin(); srcit != srcmap.end(); ++srcit, ++recv_index)
	{
		src[recv_index] = srcit->first;
		r_length[recv_index] = srcit->second;
	}
	for (int i = 0; i < num_r; ++i)
	{
		MYCALLOC(r_data[i], Barycentric *, r_length[i], sizeof(Barycentric));
	}

	com_sr_datatype(comm, num_s, num_r, dest, src, s_length, r_length, s_data, r_data, mpibaryctype, id);
	for (int i = 0; i < num_r; ++i)
	{
		for (int j = 0; j < r_length[i]; ++j)
		{
			const auto locit = baryc2locvrtxmap.find(r_data[i][j]);
			if (locit == baryc2locvrtxmap.end())
			{
				FailStreamOperation(
					"received shared barycentric vertex not found in baryc2locvrtxmap from rank " +
					std::to_string(src[i]) + ": " + DescribeBarycentricKey(r_data[i][j]));
			}
			const int locid = locit->second;
			if (newgid[locid] != -1)
			{
				FailStreamOperation("remote global id overwrite detected for local vertex " + std::to_string(locid));
			}
			newgid[locid] = r_data[i][j].newgid + globoffsets[src[i] - 1];
		}
	}

	for (auto &entry : pidmap)
	{
		delete entry.second;
	}
	for (int i = 0; i < num_r; ++i)
	{
		free(r_data[i]);
	}
	if (num_s > 0)
	{
		free(s_length);
		free(s_data);
		free(dest);
	}
	if (num_r > 0)
	{
		free(r_length);
		free(r_data);
		free(src);
	}
	free(globoffsetsml);
	free(globoffsetsmlVE);
	MPI_Type_free(&mpibaryctype);
	return newgid;
}

void WritePartitionNodesElementsFromStreams(
	const StreamMeshView &smv,
	const std::string &output_path,
	int numParts,
	int id,
	const int *newid,
	const int *VEgid,
	const std::vector<PointCoordRecord> *local_points_cache)
{
	const std::filesystem::path partition_dir =
		std::filesystem::path(output_path) / ("partitioning." + std::to_string(numParts));
	std::filesystem::create_directories(partition_dir);

	const std::filesystem::path element_path =
		partition_dir / ("part." + std::to_string(id + 1) + ".elements");
	const std::filesystem::path node_path =
		partition_dir / ("part." + std::to_string(id + 1) + ".nodes");

	std::ofstream outelements(element_path);
	if (!outelements)
	{
		FailStreamOperation("failed to open partition elements file: " + element_path.string());
	}

	constexpr int kWriterFlushLines = 4096;
	std::string element_buffer;
	element_buffer.reserve(1 << 20);
	std::ostringstream line;
	int buffered_element_lines = 0;
	const int ne = StreamMesh_GetNE(smv);
	for (int i = 1; i <= ne; ++i)
	{
		int tet[4];
		int domidx = 0;
		StreamMesh_GetVolumeElement(smv, i, tet, domidx);
		line.str("");
		line.clear();
		line << VEgid[i] << " 1 504 "
		     << newid[tet[0]] << " "
		     << newid[tet[1]] << " "
		     << newid[tet[2]] << " "
		     << newid[tet[3]] << '\n';
		element_buffer += line.str();
		if (++buffered_element_lines >= kWriterFlushLines)
		{
			outelements << element_buffer;
			element_buffer.clear();
			buffered_element_lines = 0;
		}
	}
	if (!element_buffer.empty())
	{
		outelements << element_buffer;
	}

	std::ofstream outnodes(node_path);
	if (!outnodes)
	{
		FailStreamOperation("failed to open partition nodes file: " + node_path.string());
	}

	std::string node_buffer;
	node_buffer.reserve(1 << 20);
	int buffered_node_lines = 0;
	const int np = StreamMesh_GetNP(smv);
	for (int i = 1; i <= np; ++i)
	{
		double point[3];
		if (local_points_cache != nullptr)
		{
			if (static_cast<std::size_t>(i) >= local_points_cache->size())
			{
				FailStreamOperation("local point cache index out of range in partition nodes writer");
			}
			const PointCoordRecord &cached_point = (*local_points_cache)[static_cast<std::size_t>(i)];
			point[0] = cached_point.xyz[0];
			point[1] = cached_point.xyz[1];
			point[2] = cached_point.xyz[2];
		}
		else
		{
			StreamMesh_GetPoint(smv, i, point);
		}

		line.str("");
		line.clear();
		line << newid[i] << " -1 "
		     << point[0] << " "
		     << point[1] << " "
		     << point[2] << '\n';
		node_buffer += line.str();
		if (++buffered_node_lines >= kWriterFlushLines)
		{
			outnodes << node_buffer;
			node_buffer.clear();
			buffered_node_lines = 0;
		}
	}
	if (!node_buffer.empty())
	{
		outnodes << node_buffer;
	}
}

void WritePartitionSharedFromAdjBarycs(
	const std::string &output_path,
	int numParts,
	int id,
	const int *newid,
	const std::map<int, std::list<int>> &adjbarycs)
{
	const std::filesystem::path partition_dir =
		std::filesystem::path(output_path) / ("partitioning." + std::to_string(numParts));
	std::filesystem::create_directories(partition_dir);

	const std::filesystem::path shared_path =
		partition_dir / ("part." + std::to_string(id + 1) + ".shared");
	std::ofstream out(shared_path);
	if (!out)
	{
		FailStreamOperation("failed to open partition shared file: " + shared_path.string());
	}

	constexpr int kWriterFlushLines = 4096;
	std::string shared_buffer;
	shared_buffer.reserve(1 << 20);
	std::ostringstream line;
	int buffered_shared_lines = 0;
	for (const auto &entry : adjbarycs)
	{
		const int locid = entry.first;
		std::vector<int> others(entry.second.begin(), entry.second.end());
		std::sort(others.begin(), others.end());
		others.erase(std::unique(others.begin(), others.end()), others.end());
		others.erase(std::remove(others.begin(), others.end(), id), others.end());

		std::vector<int> parts;
		parts.push_back(id);
		parts.insert(parts.end(), others.begin(), others.end());
		if (parts.size() <= 1)
		{
			continue;
		}

		line.str("");
		line.clear();
		line << newid[locid] << " " << parts.size() << " ";
		for (const int rank : parts)
		{
			line << (rank + 1) << " ";
		}
		line << '\n';
		shared_buffer += line.str();
		if (++buffered_shared_lines >= kWriterFlushLines)
		{
			out << shared_buffer;
			shared_buffer.clear();
			buffered_shared_lines = 0;
		}
	}
	if (!shared_buffer.empty())
	{
		out << shared_buffer;
	}
}

void CollectPatboundFacesFromSurfaceFile(
	const std::string &surface_file_path,
	std::set<FaceKey> &patbound_faces)
{
	std::ifstream input(surface_file_path, std::ios::binary);
	if (!input)
	{
		FailStreamOperation("failed to open surface stream file for patbound collection: " + surface_file_path);
	}

	patbound_faces.clear();
	FaceRecord record{};
	while (input.read(reinterpret_cast<char *>(&record), static_cast<std::streamsize>(sizeof(FaceRecord))))
	{
		const xdFace face = FromFaceRecord(record);
		if (face.patbound != 0)
		{
			patbound_faces.insert(make_face_key(face.lsvrtx[0], face.lsvrtx[1], face.lsvrtx[2]));
		}
	}

	if (!input.eof())
	{
		FailStreamOperation("surface stream file is truncated during patbound collection: " + surface_file_path);
	}
}

StreamBoundaryHeaderStats WritePartitionBoundaryAndHeaderFromStreams(
	const StreamMeshView &smv,
	const std::string &output_path,
	int numParts,
	int id,
	const int *newid,
	const int *VEgid,
	const std::map<int, std::list<int>> &adjbarycs,
	bool keep_shard_files)
{
	constexpr std::size_t kBoundaryShardFlushRecords = 4096;
	constexpr std::size_t kBoundaryShardCount = 256;
	constexpr int kBoundaryWriterFlushLines = 4096;
	StreamBoundaryHeaderStats stats;

	auto write_tet_face_records = [](std::ofstream &output,
		const std::vector<BoundaryTetFaceRecord> &records)
	{
		if (records.empty())
		{
			return;
		}
		output.write(reinterpret_cast<const char *>(records.data()),
			static_cast<std::streamsize>(records.size() * sizeof(BoundaryTetFaceRecord)));
		if (!output)
		{
			FailStreamOperation("failed to write tet-face shard records");
		}
	};

	auto write_surface_records = [](std::ofstream &output,
		const std::vector<BoundarySurfaceRecord> &records)
	{
		if (records.empty())
		{
			return;
		}
		output.write(reinterpret_cast<const char *>(records.data()),
			static_cast<std::streamsize>(records.size() * sizeof(BoundarySurfaceRecord)));
		if (!output)
		{
			FailStreamOperation("failed to write boundary-surface shard records");
		}
	};

	auto build_tet_face_shard_files = [&](const std::filesystem::path &work_dir)
	{
		std::vector<std::string> shard_files(kBoundaryShardCount);
		std::vector<std::ofstream> outputs(kBoundaryShardCount);
		std::vector<std::vector<BoundaryTetFaceRecord>> buffers(kBoundaryShardCount);
		for (std::size_t shard = 0; shard < kBoundaryShardCount; ++shard)
		{
			const std::filesystem::path filepath =
				work_dir / ("boundary_tetface_shard_" + std::to_string(shard) + ".bin");
			shard_files[shard] = filepath.string();
			EnsureParentDirectory(shard_files[shard]);
			outputs[shard].open(filepath, std::ios::binary | std::ios::trunc);
			if (!outputs[shard])
			{
				FailStreamOperation("failed to open tet-face shard file: " + filepath.string());
			}
			buffers[shard].reserve(kBoundaryShardFlushRecords);
		}

		std::ifstream input(smv.tet_file_path, std::ios::binary);
		if (!input)
		{
			FailStreamOperation("failed to open tet stream for boundary sharding: " + smv.tet_file_path);
		}

		TetRecord record{};
		int eid = 0;
		while (input.read(reinterpret_cast<char *>(&record), static_cast<std::streamsize>(sizeof(TetRecord))))
		{
			++eid;
			int tet[4];
			int domidx = 0;
			FromTetRecord(record, tet, domidx);
			for (int j = 0; j < 4; ++j)
			{
				int face_vids[3];
				int l = 0;
				for (int k = 0; k < 4; ++k)
				{
					if (k != j)
					{
						face_vids[l] = newid[tet[k]];
						++l;
					}
				}
				const FaceKey key = make_face_key(face_vids[0], face_vids[1], face_vids[2]);
				const std::size_t shard = BoundaryFaceShardIndex(key, kBoundaryShardCount);
				buffers[shard].push_back(BoundaryTetFaceRecord{key.v[0], key.v[1], key.v[2], VEgid[eid]});
				if (buffers[shard].size() >= kBoundaryShardFlushRecords)
				{
					write_tet_face_records(outputs[shard], buffers[shard]);
					buffers[shard].clear();
				}
			}
		}

		if (!input.eof())
		{
			FailStreamOperation("tet stream is truncated during boundary sharding: " + smv.tet_file_path);
		}

		for (std::size_t shard = 0; shard < kBoundaryShardCount; ++shard)
		{
			write_tet_face_records(outputs[shard], buffers[shard]);
		}
		return shard_files;
	};

	auto build_surface_shard_files = [&](const std::filesystem::path &work_dir)
	{
		std::vector<std::string> shard_files(kBoundaryShardCount);
		std::vector<std::ofstream> outputs(kBoundaryShardCount);
		std::vector<std::vector<BoundarySurfaceRecord>> buffers(kBoundaryShardCount);
		for (std::size_t shard = 0; shard < kBoundaryShardCount; ++shard)
		{
			const std::filesystem::path filepath =
				work_dir / ("boundary_surface_shard_" + std::to_string(shard) + ".bin");
			shard_files[shard] = filepath.string();
			EnsureParentDirectory(shard_files[shard]);
			outputs[shard].open(filepath, std::ios::binary | std::ios::trunc);
			if (!outputs[shard])
			{
				FailStreamOperation("failed to open boundary-surface shard file: " + filepath.string());
			}
			buffers[shard].reserve(kBoundaryShardFlushRecords);
		}

		std::ifstream input(smv.surface_file_path, std::ios::binary);
		if (!input)
		{
			FailStreamOperation("failed to open surface stream for boundary sharding: " + smv.surface_file_path);
		}

		FaceRecord record{};
		while (input.read(reinterpret_cast<char *>(&record), static_cast<std::streamsize>(sizeof(FaceRecord))))
		{
			const xdFace face = FromFaceRecord(record);
			if (face.patbound != 0)
			{
				++stats.skipped_patbound;
				continue;
			}

			const int gv0 = newid[face.lsvrtx[0]];
			const int gv1 = newid[face.lsvrtx[1]];
			const int gv2 = newid[face.lsvrtx[2]];
			const FaceKey key = make_face_key(gv0, gv1, gv2);
			const std::size_t shard = BoundaryFaceShardIndex(key, kBoundaryShardCount);
			buffers[shard].push_back(BoundarySurfaceRecord{
				key.v[0],
				key.v[1],
				key.v[2],
				face.geoboundary + 1,
				gv0,
				gv1,
				gv2});
			if (buffers[shard].size() >= kBoundaryShardFlushRecords)
			{
				write_surface_records(outputs[shard], buffers[shard]);
				buffers[shard].clear();
			}
		}

		if (!input.eof())
		{
			FailStreamOperation("surface stream is truncated during boundary sharding: " + smv.surface_file_path);
		}

		for (std::size_t shard = 0; shard < kBoundaryShardCount; ++shard)
		{
			write_surface_records(outputs[shard], buffers[shard]);
		}
		return shard_files;
	};

	auto write_boundary_from_shards = [&](const std::vector<std::string> &tet_face_shards,
		const std::vector<std::string> &surface_shards,
		const std::filesystem::path &boundary_path)
	{
		std::ofstream outboundarys(boundary_path);
		if (!outboundarys)
		{
			FailStreamOperation("failed to open partition boundary file: " + boundary_path.string());
		}

		std::string boundary_buffer;
		boundary_buffer.reserve(1 << 20);
		std::ostringstream line;
		int buffered_boundary_lines = 0;
		int number = 0;

		for (std::size_t shard = 0; shard < tet_face_shards.size(); ++shard)
		{
			const int record_count =
				CountRecordsByFileSize(tet_face_shards[shard], sizeof(BoundaryTetFaceRecord));
			std::unordered_map<FaceKey, int, FaceKeyHash> face2vol;
			face2vol.reserve(static_cast<std::size_t>(record_count) * 2 + 1024);

			std::ifstream tet_input(tet_face_shards[shard], std::ios::binary);
			if (!tet_input)
			{
				FailStreamOperation("failed to open tet-face shard file: " + tet_face_shards[shard]);
			}

			const double build_face2vol_t0 = MPI_Wtime();
			BoundaryTetFaceRecord tet_record{};
			while (tet_input.read(reinterpret_cast<char *>(&tet_record),
				static_cast<std::streamsize>(sizeof(BoundaryTetFaceRecord))))
			{
				face2vol.emplace(FaceKey{{tet_record.a, tet_record.b, tet_record.c}}, tet_record.vegid);
			}
			if (!tet_input.eof())
			{
				FailStreamOperation("tet-face shard file is truncated: " + tet_face_shards[shard]);
			}
			stats.time_build_face2vol += MPI_Wtime() - build_face2vol_t0;
			stats.face2vol_size = std::max(stats.face2vol_size, face2vol.size());

			std::ifstream surface_input(surface_shards[shard], std::ios::binary);
			if (!surface_input)
			{
				FailStreamOperation("failed to open boundary-surface shard file: " + surface_shards[shard]);
			}

			const double scan_surface_t0 = MPI_Wtime();
			BoundarySurfaceRecord surface_record{};
			while (surface_input.read(reinterpret_cast<char *>(&surface_record),
				static_cast<std::streamsize>(sizeof(BoundarySurfaceRecord))))
			{
				const auto it = face2vol.find(FaceKey{{surface_record.key_a, surface_record.key_b, surface_record.key_c}});
				if (it == face2vol.end())
				{
					++stats.missed_surface_faces;
					continue;
				}

				++stats.matched_surface_faces;
				++number;
				line.str("");
				line.clear();
				line << number << " " << surface_record.geoid << " " << it->second
				     << " 0 303 "
				     << surface_record.v0 << " "
				     << surface_record.v1 << " "
				     << surface_record.v2 << '\n';
				boundary_buffer += line.str();
				if (++buffered_boundary_lines >= kBoundaryWriterFlushLines)
				{
					outboundarys << boundary_buffer;
					boundary_buffer.clear();
					buffered_boundary_lines = 0;
				}
			}
			if (!surface_input.eof())
			{
				FailStreamOperation("boundary-surface shard file is truncated: " + surface_shards[shard]);
			}
			stats.time_scan_surface += MPI_Wtime() - scan_surface_t0;
		}

		if (!boundary_buffer.empty())
		{
			outboundarys << boundary_buffer;
		}
		return number;
	};

	const std::filesystem::path partition_dir =
		std::filesystem::path(output_path) / ("partitioning." + std::to_string(numParts));
	std::filesystem::create_directories(partition_dir);

	const std::filesystem::path boundary_path =
		partition_dir / ("part." + std::to_string(id + 1) + ".boundary");
	const std::filesystem::path header_path =
		partition_dir / ("part." + std::to_string(id + 1) + ".header");
	const std::filesystem::path work_dir =
		partition_dir / ("boundary_shards_rank" + std::to_string(id + 1));

	const int np = StreamMesh_GetNP(smv);
	const int ne = StreamMesh_GetNE(smv);

	const double build_tet_face_shards_t0 = MPI_Wtime();
	const std::vector<std::string> tet_face_shards = build_tet_face_shard_files(work_dir);
	stats.time_build_face2vol += MPI_Wtime() - build_tet_face_shards_t0;
	PrintStreamStageMem("after BuildTetFaceShardFiles", output_path);

	const double build_surface_shards_t0 = MPI_Wtime();
	const std::vector<std::string> surface_shards = build_surface_shard_files(work_dir);
	stats.time_scan_surface += MPI_Wtime() - build_surface_shards_t0;
	PrintStreamStageMem("after BuildBoundarySurfaceShardFiles", output_path);

	const int number = write_boundary_from_shards(tet_face_shards, surface_shards, boundary_path);
	PrintStreamStageMem("after shard boundary join", output_path);

	if (!keep_shard_files)
	{
		std::error_code ec;
		std::filesystem::remove_all(work_dir, ec);
	}

	const double write_header_t0 = MPI_Wtime();
	std::ofstream outheader(header_path);
	if (!outheader)
	{
		FailStreamOperation("failed to open partition header file: " + header_path.string());
	}

	outheader << np << " " << ne << " " << number << '\n';
	outheader << 2 << '\n';
	outheader << "504 " << ne << '\n';
	outheader << "303 " << number << '\n';
	if (!adjbarycs.empty())
	{
		outheader << adjbarycs.size() << " 0" << '\n';
	}
	stats.time_write_header = MPI_Wtime() - write_header_t0;

	stats.boundary_count = number;
	return stats;
}

StreamMeshQualityStats ComputeMeshQualityFromStreams(const StreamMeshView &smv)
{
	StreamMeshQualityStats stats;
	stats.ne = StreamMesh_GetNE(smv);
	if (stats.ne == 0)
	{
		return stats;
	}

	double sum_volume = 0.0;
	stats.min_volume = std::numeric_limits<double>::max();
	stats.max_volume = 0.0;

	for (int i = 1; i <= stats.ne; ++i)
	{
		int tet[4];
		int domidx = 0;
		double p0[3];
		double p1[3];
		double p2[3];
		double p3[3];
		StreamMesh_GetVolumeElement(smv, i, tet, domidx);
		StreamMesh_GetPoint(smv, tet[0], p0);
		StreamMesh_GetPoint(smv, tet[1], p1);
		StreamMesh_GetPoint(smv, tet[2], p2);
		StreamMesh_GetPoint(smv, tet[3], p3);

		const double volume = TetVolumeFromPoints(p0, p1, p2, p3);
		if (volume < stats.min_volume)
		{
			stats.min_volume = volume;
		}
		if (volume > stats.max_volume)
		{
			stats.max_volume = volume;
		}
		sum_volume += volume;
	}

	stats.avg_volume = sum_volume / static_cast<double>(stats.ne);
	return stats;
}

void WriteMeshQualityFromStreams(
	const std::string &output_path,
	int id,
	const StreamMeshQualityStats &stats)
{
	const std::filesystem::path meshquality_dir =
		std::filesystem::path(output_path) / "meshQuality";
	std::filesystem::create_directories(meshquality_dir);

	const std::filesystem::path filepath =
		meshquality_dir / ("meshQuality" + std::to_string(id) + ".txt");
	std::ofstream output(filepath);
	if (!output)
	{
		FailStreamOperation("failed to open meshQuality file: " + filepath.string());
	}

	output << std::setprecision(17);
	output << "ne=" << stats.ne << "\n";
	output << "min_volume=" << stats.min_volume << "\n";
	output << "max_volume=" << stats.max_volume << "\n";
	output << "avg_volume=" << stats.avg_volume << "\n";
}

static void BuildLocalPointCacheForVolAdj(const StreamMeshView &smv,
	const int *newid,
	StreamVolWithAdjData &out_data)
{
	const int local_np = StreamMesh_GetNP(smv);
	out_data.local_points_cache.clear();
	out_data.local_points_cache.resize(static_cast<std::size_t>(local_np + 1));
	out_data.local_global_pid_to_local_pid.clear();

	int max_global_pid = 0;
	for (int pid = 1; pid <= local_np; ++pid)
	{
		PointCoordRecord &point = out_data.local_points_cache[static_cast<std::size_t>(pid)];
		StreamMesh_GetPoint(smv, pid, point.xyz);
		if (newid[pid] > 0)
		{
			out_data.local_global_pid_to_local_pid[newid[pid]] = pid;
			max_global_pid = (std::max)(max_global_pid, newid[pid]);
		}
	}

	out_data.global_pid_to_local_pid_dense.clear();
	if (max_global_pid > 0 &&
		max_global_pid <= 2000000 &&
		max_global_pid <= local_np * 2)
	{
		out_data.global_pid_to_local_pid_dense.assign(static_cast<std::size_t>(max_global_pid + 1), 0);
		for (const auto &entry : out_data.local_global_pid_to_local_pid)
		{
			out_data.global_pid_to_local_pid_dense[static_cast<std::size_t>(entry.first)] = entry.second;
		}
	}
}

static void LookupVolAdjPointXYZ(const StreamVolWithAdjData &data,
	const int *newid,
	int global_pid,
	double xyz[3])
{
	if (data.smv == nullptr)
	{
		FailStreamOperation("volwithadj lookup requested without stream mesh view");
	}

	int local_pid = 0;
	if (!data.global_pid_to_local_pid_dense.empty() &&
		global_pid >= 0 &&
		static_cast<std::size_t>(global_pid) < data.global_pid_to_local_pid_dense.size())
	{
		local_pid = data.global_pid_to_local_pid_dense[static_cast<std::size_t>(global_pid)];
	}
	if (local_pid == 0)
	{
		const auto local_it = data.local_global_pid_to_local_pid.find(global_pid);
		if (local_it != data.local_global_pid_to_local_pid.end())
		{
			local_pid = local_it->second;
		}
	}
	if (local_pid > 0)
	{
		if (static_cast<std::size_t>(local_pid) >= data.local_points_cache.size())
		{
			FailStreamOperation("local point cache index out of range for volwithadj point lookup");
		}
		const PointCoordRecord &point = data.local_points_cache[static_cast<std::size_t>(local_pid)];
		xyz[0] = point.xyz[0];
		xyz[1] = point.xyz[1];
		xyz[2] = point.xyz[2];
		return;
	}

	const auto ghost_it = data.ghost_global_pid_to_index.find(global_pid);
	if (ghost_it != data.ghost_global_pid_to_index.end())
	{
		const PointCoordRecord &point = data.ghost_points.at(static_cast<std::size_t>(ghost_it->second));
		xyz[0] = point.xyz[0];
		xyz[1] = point.xyz[1];
		xyz[2] = point.xyz[2];
		return;
	}

	FailStreamOperation("failed to locate point for volwithadj global point id: " + std::to_string(global_pid));
}

StreamFullMeshQualityStats ComputeFullMeshQualityFromVolWithAdjStreams(
	const StreamVolWithAdjData &data,
	const int *newid,
	const std::string &output_path)
{
	if (data.smv == nullptr)
	{
		FailStreamOperation("full meshQuality requested without stream mesh view");
	}

	const StreamMeshView &smv = *data.smv;
	StreamFullMeshQualityStats stats;
	stats.np = StreamMesh_GetNP(smv) + static_cast<int>(data.ghost_points.size());
	stats.nse = StreamMesh_GetNSE(smv);
	stats.ne = StreamMesh_GetNE(smv) + static_cast<int>(data.ghost_tets.size());
	stats.triangle_length_width_ratio_min = 0x3f3f3f;
	stats.triangle_skew_min = 0x3f3f3f;
	stats.tetrahedrons_length_width_ratio_min = 0x3f3f3f;
	stats.tetrahedrons_skew_min = 0x3f3f3f;
	stats.min_volume = std::numeric_limits<double>::max();

	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	AppendRankTestoutLine(
		output_path,
		rank,
		"[STREAM] meshQuality surface pass begin rank=" + std::to_string(rank));
	std::ifstream surface_input(smv.surface_file_path, std::ios::binary);
	if (!surface_input)
	{
		FailStreamOperation("failed to open surface stream file for full meshQuality: " + smv.surface_file_path);
	}
	std::vector<FaceRecord> face_buffer;
	while (ReadFaceBatch(surface_input, face_buffer, kDefaultStreamBatch) > 0)
	{
		for (const FaceRecord &record : face_buffer)
		{
			const xdFace face = FromFaceRecord(record);
			double xyz[3][3];
			for (int k = 0; k < 3; ++k)
			{
				const PointCoordRecord &point =
					data.local_points_cache.at(static_cast<std::size_t>(face.lsvrtx[k]));
				xyz[k][0] = point.xyz[0];
				xyz[k][1] = point.xyz[1];
				xyz[k][2] = point.xyz[2];
			}

			const double lwr = length_width_ratio(xyz);
			Record_LWR_count(lwr, stats.aspect_ratio_count);
			stats.triangle_length_width_ratio_sum += lwr;
			stats.triangle_length_width_ratio_min = (std::min)(stats.triangle_length_width_ratio_min, lwr);
			stats.triangle_length_width_ratio_max = (std::max)(stats.triangle_length_width_ratio_max, lwr);

			const double min_ia = min_internal_angle(xyz);
			const double max_ia = max_internal_angle(xyz);
			stats.triangle_internal_angle_min = min_ia;
			stats.triangle_internal_angle_max = max_ia;

			const double tris = triangle_skew(xyz);
			stats.triangle_skew_sum += tris;
			stats.triangle_skew_min = (std::min)(stats.triangle_skew_min, tris);
			stats.triangle_skew_max = (std::max)(stats.triangle_skew_max, tris);
		}
	}

	auto update_tet_quality = [&](const double Vxyz[4][3]) {
		const double vlwr = tetrahedrons_length_width_ratio(Vxyz);
		stats.tetrahedrons_length_width_ratio_sum += vlwr;
		stats.tetrahedrons_length_width_ratio_min = (std::min)(stats.tetrahedrons_length_width_ratio_min, vlwr);
		stats.tetrahedrons_length_width_ratio_max = (std::max)(stats.tetrahedrons_length_width_ratio_max, vlwr);

		const double vtris = tetrahedrons_skew(const_cast<double (*)[3]>(Vxyz));
		stats.tetrahedrons_skew_sum += vtris;
		stats.tetrahedrons_skew_min = (std::min)(stats.tetrahedrons_skew_min, vtris);
		stats.tetrahedrons_skew_max = (std::max)(stats.tetrahedrons_skew_max, vtris);

		double face_xyz[4][3][3];
		const int u[4][3] = {
			{0, 1, 2},
			{0, 1, 3},
			{0, 2, 3},
			{1, 2, 3},
		};
		for (int k = 0; k < 4; ++k)
		{
			for (int j = 0; j < 3; ++j)
			{
				face_xyz[k][j][0] = Vxyz[u[k][j]][0];
				face_xyz[k][j][1] = Vxyz[u[k][j]][1];
				face_xyz[k][j][2] = Vxyz[u[k][j]][2];
			}
		}
		double face_VMinIA[4];
		double face_VMaxIA[4];
		for (int k = 0; k < 4; ++k)
		{
			face_VMinIA[k] = min_internal_angle(face_xyz[k]);
			face_VMaxIA[k] = max_internal_angle(face_xyz[k]);
		}
		stats.tetrahedrons_internal_angle_min =
			(std::min)({face_VMinIA[0], face_VMinIA[1], face_VMinIA[2], face_VMinIA[3]});
		stats.tetrahedrons_internal_angle_max =
			(std::max)({face_VMaxIA[0], face_VMaxIA[1], face_VMaxIA[2], face_VMaxIA[3]});

		const double volume = TetVolumeFromPoints(Vxyz[0], Vxyz[1], Vxyz[2], Vxyz[3]);
		stats.min_volume = (std::min)(stats.min_volume, volume);
		stats.max_volume = (std::max)(stats.max_volume, volume);
		stats.volume_sum += volume;
	};

	AppendRankTestoutLine(
		output_path,
		rank,
		"[STREAM] meshQuality volume pass begin rank=" + std::to_string(rank));
	std::ifstream tet_input(smv.tet_file_path, std::ios::binary);
	if (!tet_input)
	{
		FailStreamOperation("failed to open tet stream file for full meshQuality: " + smv.tet_file_path);
	}
	std::vector<TetRecord> tet_buffer;
	while (ReadTetBatch(tet_input, tet_buffer, kDefaultStreamBatch) > 0)
	{
		for (const TetRecord &record : tet_buffer)
		{
			int tet[4];
			int domidx = 0;
			double Vxyz[4][3];
			FromTetRecord(record, tet, domidx);
			for (int k = 0; k < 4; ++k)
			{
				const PointCoordRecord &point =
					data.local_points_cache.at(static_cast<std::size_t>(tet[k]));
				Vxyz[k][0] = point.xyz[0];
				Vxyz[k][1] = point.xyz[1];
				Vxyz[k][2] = point.xyz[2];
			}
			update_tet_quality(Vxyz);
		}
	}

	for (const StreamGhostTetRecord &record : data.ghost_tets)
	{
		double Vxyz[4][3];
		for (int k = 0; k < 4; ++k)
		{
			LookupVolAdjPointXYZ(data, newid, record.global_vids[k], Vxyz[k]);
		}
		update_tet_quality(Vxyz);
	}

	if (stats.nse == 0)
	{
		stats.triangle_length_width_ratio_min = 0.0;
		stats.triangle_skew_min = 0.0;
	}
	if (stats.ne == 0)
	{
		stats.tetrahedrons_length_width_ratio_min = 0.0;
		stats.tetrahedrons_skew_min = 0.0;
		stats.min_volume = 0.0;
	}

	return stats;
}

void WriteFullMeshQualityFromVolWithAdjStreams(
	const std::string &output_path,
	int id,
	const StreamFullMeshQualityStats &stats,
	MPI_Comm comm)
{
	const std::filesystem::path meshquality_dir =
		std::filesystem::path(output_path) / "meshQuality";
	std::filesystem::create_directories(meshquality_dir);

	int global_aspect_ratio_count[6] = {0, 0, 0, 0, 0, 0};
	MPI_Reduce(
		const_cast<int *>(stats.aspect_ratio_count),
		global_aspect_ratio_count,
		6,
		MPI_INT,
		MPI_SUM,
		0,
		comm);

	if (id == 0)
	{
		const std::filesystem::path summary_path = meshquality_dir / "meshQuality.txt";
		FILE *summary_fp = std::fopen(summary_path.string().c_str(), "w");
		if (summary_fp == NULL)
		{
			FailStreamOperation("failed to open meshQuality summary file: " + summary_path.string());
		}
		int sum_count_surface = 0;
		for (int i = 0; i < 6; ++i)
		{
			sum_count_surface += global_aspect_ratio_count[i];
		}
		const double denom = (sum_count_surface == 0) ? 1.0 : static_cast<double>(sum_count_surface);
		fprintf(summary_fp, "Sum_Aspect_Ratio(1-1.5): %f \r\n", global_aspect_ratio_count[0] / denom);
		fprintf(summary_fp, "Sum_Aspect_Ratio(1.5-2): %f \r\n", global_aspect_ratio_count[1] / denom);
		fprintf(summary_fp, "Sum_Aspect_Ratio(2-3): %f \r\n", global_aspect_ratio_count[2] / denom);
		fprintf(summary_fp, "Sum_Aspect_Ratio(3-4): %f \r\n", global_aspect_ratio_count[3] / denom);
		fprintf(summary_fp, "Sum_Aspect_Ratio(4-5): %f \r\n", global_aspect_ratio_count[4] / denom);
		fprintf(summary_fp, "Sum_Aspect_Ratio(5-6): %f \r\n", global_aspect_ratio_count[5] / denom);
		std::fclose(summary_fp);
	}

	const std::filesystem::path rank_path =
		meshquality_dir / ("meshQuality" + std::to_string(id) + ".txt");
	FILE *fp = std::fopen(rank_path.string().c_str(), "w");
	if (fp == NULL)
	{
		FailStreamOperation("failed to open rank meshQuality file: " + rank_path.string());
	}

	const double nse_denom = (stats.nse == 0) ? 1.0 : static_cast<double>(stats.nse);
	const double ne_denom = (stats.ne == 0) ? 1.0 : static_cast<double>(stats.ne);
	fprintf(fp, "Point_Num: %d SurfEle_Num: %d SoildEle_Num: %d \r\n", stats.np, stats.nse, stats.ne);
	fprintf(fp, "\r\n");
	fprintf(fp, "triangle_length_width_ratio_min: %f \r\n", stats.triangle_length_width_ratio_min);
	fprintf(fp, "triangle_length_width_ratio_max: %f \r\n", stats.triangle_length_width_ratio_max);
	fprintf(fp, "triangle_length_width_ratio_mean: %f \r\n", stats.triangle_length_width_ratio_sum / nse_denom);
	fprintf(fp, "Aspect_Ratio(1-1.5): %d \r\n", stats.aspect_ratio_count[0]);
	fprintf(fp, "Aspect_Ratio(1.5-2): %d \r\n", stats.aspect_ratio_count[1]);
	fprintf(fp, "Aspect_Ratio(2-3): %d \r\n", stats.aspect_ratio_count[2]);
	fprintf(fp, "Aspect_Ratio(3-4): %d \r\n", stats.aspect_ratio_count[3]);
	fprintf(fp, "Aspect_Ratio(4-5): %d \r\n", stats.aspect_ratio_count[4]);
	fprintf(fp, "Aspect_Ratio(5-6): %d \r\n", stats.aspect_ratio_count[5]);
	fprintf(fp, "triangle_skew_min: %f \r\n", stats.triangle_skew_min);
	fprintf(fp, "triangle_skew_max: %f \r\n", stats.triangle_skew_max);
	fprintf(fp, "triangle_skew_mean: %f \r\n", stats.triangle_skew_sum / nse_denom);
	fprintf(fp, "triangle_internal_angle_min: %f \r\n", stats.triangle_internal_angle_min);
	fprintf(fp, "triangle_internal_angle_max: %f \r(n", stats.triangle_internal_angle_max);
	fprintf(fp, "\r\n");
	fprintf(fp, "tetrahedrons_length_width_ratio_min: %f \r\n", stats.tetrahedrons_length_width_ratio_min);
	fprintf(fp, "tetrahedrons_length_width_ratio_max: %f \r\n", stats.tetrahedrons_length_width_ratio_max);
	fprintf(fp, "tetrahedrons_length_width_ratio_mean: %f \r\n", stats.tetrahedrons_length_width_ratio_sum / ne_denom);
	fprintf(fp, "tetrahedrons_skew_min: %f \r\n", stats.tetrahedrons_skew_min);
	fprintf(fp, "tetrahedrons_skew_max: %f \r\n", stats.tetrahedrons_skew_max);
	fprintf(fp, "tetrahedrons_skew_mean: %f \r\n", stats.tetrahedrons_skew_sum / ne_denom);
	fprintf(fp, "tetrahedrons_internal_angle_min: %f \r\n", stats.tetrahedrons_internal_angle_min);
	fprintf(fp, "tetrahedrons_internal_angle_max: %f \r\n", stats.tetrahedrons_internal_angle_max);
	std::fclose(fp);
}

void com_baryVolumeElements_from_streams(
	const StreamMeshView &smv,
	MPI_Comm comm,
	const std::map<int, std::list<int>> &adjbarycs,
	const int *newid,
	const int *VEgid,
	int numprocs,
	int mypid,
	StreamVolWithAdjData &out_data)
{
	out_data.smv = &smv;
	out_data.ghost_tets.clear();
	out_data.ghost_point_global_ids.clear();
	out_data.ghost_points.clear();
	out_data.ghost_global_pid_to_index.clear();
	BuildLocalPointCacheForVolAdj(smv, newid, out_data);

	const int ne = StreamMesh_GetNE(smv);

	std::map<int, std::vector<StreamGhostVE>> send_map;
	for (int eid = 1; eid <= ne; ++eid)
	{
		int tet[4];
		int domidx = 0;
		StreamMesh_GetVolumeElement(smv, eid, tet, domidx);

		std::map<int, int> num_adj_points;
		for (int k = 0; k < 4; ++k)
		{
			const auto ib = adjbarycs.find(tet[k]);
			if (ib == adjbarycs.end())
			{
				continue;
			}
			for (const int pid : ib->second)
			{
				num_adj_points[pid]++;
			}
		}

		std::set<int> pid_tmp;
		for (const auto &entry : num_adj_points)
		{
			if (entry.second >= 3)
			{
				pid_tmp.insert(entry.first);
			}
		}

		if (pid_tmp.empty())
		{
			continue;
		}

		StreamGhostVE ve{};
		ve.gid = VEgid[eid];
		ve.domidx = domidx;
		for (int k = 0; k < 4; ++k)
		{
			ve.pindex[k] = newid[tet[k]];
			const PointCoordRecord &point =
				out_data.local_points_cache.at(static_cast<std::size_t>(tet[k]));
			ve.xyz[k][0] = point.xyz[0];
			ve.xyz[k][1] = point.xyz[1];
			ve.xyz[k][2] = point.xyz[2];
		}

		for (const int dst : pid_tmp)
		{
			send_map[dst].push_back(ve);
		}
	}

	std::vector<int> send_counts(numprocs, 0);
	for (const auto &entry : send_map)
	{
		if (entry.first >= 0 && entry.first < numprocs)
		{
			send_counts[entry.first] = static_cast<int>(entry.second.size());
		}
	}

	std::vector<int> recv_counts(numprocs, 0);
	MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

	std::vector<int> send_displs(numprocs, 0);
	std::vector<int> recv_displs(numprocs, 0);
	int total_send = 0;
	int total_recv = 0;
	for (int rank = 0; rank < numprocs; ++rank)
	{
		send_displs[rank] = total_send;
		recv_displs[rank] = total_recv;
		total_send += send_counts[rank];
		total_recv += recv_counts[rank];
	}

	std::vector<StreamGhostVE> send_buffer;
	send_buffer.reserve(total_send);
	for (int rank = 0; rank < numprocs; ++rank)
	{
		const auto it = send_map.find(rank);
		if (it != send_map.end())
		{
			send_buffer.insert(send_buffer.end(), it->second.begin(), it->second.end());
		}
	}

	std::vector<StreamGhostVE> recv_buffer(static_cast<std::size_t>(total_recv));
	MPI_Datatype ghost_ve_type = CreateMpiGhostVeType();
	MPI_Alltoallv(
		send_buffer.empty() ? nullptr : send_buffer.data(),
		send_counts.data(),
		send_displs.data(),
		ghost_ve_type,
		recv_buffer.empty() ? nullptr : recv_buffer.data(),
		recv_counts.data(),
		recv_displs.data(),
		ghost_ve_type,
		comm);
	MPI_Type_free(&ghost_ve_type);

	out_data.ghost_tets.reserve(recv_buffer.size());
	for (const StreamGhostVE &ve : recv_buffer)
	{
		StreamGhostTetRecord ghost_record{};
		ghost_record.gid = ve.gid;
		ghost_record.domidx = ve.domidx;
		for (int k = 0; k < 4; ++k)
		{
			ghost_record.global_vids[k] = ve.pindex[k];
			if (out_data.local_global_pid_to_local_pid.find(ve.pindex[k]) != out_data.local_global_pid_to_local_pid.end())
			{
				continue;
			}
			if (out_data.ghost_global_pid_to_index.find(ve.pindex[k]) == out_data.ghost_global_pid_to_index.end())
			{
				PointCoordRecord point{};
				point.xyz[0] = ve.xyz[k][0];
				point.xyz[1] = ve.xyz[k][1];
				point.xyz[2] = ve.xyz[k][2];
				out_data.ghost_point_global_ids.push_back(ve.pindex[k]);
				out_data.ghost_points.push_back(point);
				out_data.ghost_global_pid_to_index[ve.pindex[k]] =
					static_cast<int>(out_data.ghost_points.size() - 1);
			}
		}
		out_data.ghost_tets.push_back(ghost_record);
	}
}

void WriteVolWithAdjFromStreams(const std::string &output_path,
	int id,
	const StreamVolWithAdjData &data,
	const int *newid)
{
	if (data.smv == nullptr)
	{
		FailStreamOperation("volwithadj write requested without stream mesh view");
	}

	const StreamMeshView &smv = *data.smv;
	const int local_np = StreamMesh_GetNP(smv);
	const int local_ne = StreamMesh_GetNE(smv);
	const int nse = StreamMesh_GetNSE(smv);
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	AppendRankTestoutLine(
		output_path,
		rank,
		"[STREAM] volwithadj writer begin rank=" + std::to_string(rank));
	const std::filesystem::path out_path =
		std::filesystem::path(output_path) / "volwithadj" / ("volwithadj" + std::to_string(id) + ".vol");
	EnsureParentDirectory(out_path.string());
	std::ofstream output(out_path, std::ios::trunc);
	if (!output)
	{
		FailStreamOperation("failed to open volwithadj output file: " + out_path.string());
	}

	std::map<int, int> writer_index_by_global_id;
	for (int pid = 1; pid <= local_np; ++pid)
	{
		writer_index_by_global_id[newid[pid]] = pid;
	}
	for (std::size_t i = 0; i < data.ghost_point_global_ids.size(); ++i)
	{
		writer_index_by_global_id[data.ghost_point_global_ids[i]] = local_np + static_cast<int>(i) + 1;
	}

	std::ifstream surface_scan(smv.surface_file_path, std::ios::binary);
	if (!surface_scan)
	{
		FailStreamOperation("failed to open surface stream file for volwithadj descriptors: " + smv.surface_file_path);
	}
	std::vector<FaceRecord> face_buffer;
	int max_geoboundary = 0;
	while (ReadFaceBatch(surface_scan, face_buffer, kDefaultStreamBatch) > 0)
	{
		for (const FaceRecord &record : face_buffer)
		{
			max_geoboundary = std::max(max_geoboundary, std::max(record.geoboundary, 1));
		}
	}

	output << "# Generated by NETGEN stream volwithadj writer\n\n";
	output << "mesh3d\n";
	output << "dimension\n3\n";
	output << "geomtype\n0\n\n";
	output << "# surfnr\tdomin\tdomout\ttlosurf\tbcprop\n";
	output << "facedescriptors\n";
	output << max_geoboundary << "\n";
	for (int descriptor = 1; descriptor <= max_geoboundary; ++descriptor)
	{
		output << descriptor << " 1 0 " << descriptor << " " << descriptor << "\n";
	}

	output << "surfaceelements\n";
	output << nse << "\n";
	std::ifstream surface_input(smv.surface_file_path, std::ios::binary);
	if (!surface_input)
	{
		FailStreamOperation("failed to open surface stream file for volwithadj write: " + smv.surface_file_path);
	}
	while (ReadFaceBatch(surface_input, face_buffer, kDefaultStreamBatch) > 0)
	{
		for (const FaceRecord &record : face_buffer)
		{
			const int descriptor = std::max(record.geoboundary, 1);
			output << " " << descriptor << " 1 1 0 3 "
			       << writer_index_by_global_id.at(newid[record.lsvrtx[0]]) << " "
			       << writer_index_by_global_id.at(newid[record.lsvrtx[1]]) << " "
			       << writer_index_by_global_id.at(newid[record.lsvrtx[2]]) << "\n";
		}
	}

	output << "volumeelements\n";
	output << (local_ne + static_cast<int>(data.ghost_tets.size())) << "\n";
	std::ifstream tet_input(smv.tet_file_path, std::ios::binary);
	if (!tet_input)
	{
		FailStreamOperation("failed to open tet stream file for volwithadj write: " + smv.tet_file_path);
	}
	std::vector<TetRecord> tet_buffer;
	while (ReadTetBatch(tet_input, tet_buffer, kDefaultStreamBatch) > 0)
	{
		for (const TetRecord &record : tet_buffer)
		{
			int tet[4];
			int domidx = 0;
			FromTetRecord(record, tet, domidx);
			output << domidx << " 4 "
			       << writer_index_by_global_id.at(newid[tet[0]]) << " "
			       << writer_index_by_global_id.at(newid[tet[1]]) << " "
			       << writer_index_by_global_id.at(newid[tet[2]]) << " "
			       << writer_index_by_global_id.at(newid[tet[3]]) << "\n";
		}
	}
	for (const StreamGhostTetRecord &record : data.ghost_tets)
	{
		output << record.domidx << " 4 "
		       << writer_index_by_global_id.at(record.global_vids[0]) << " "
		       << writer_index_by_global_id.at(record.global_vids[1]) << " "
		       << writer_index_by_global_id.at(record.global_vids[2]) << " "
		       << writer_index_by_global_id.at(record.global_vids[3]) << "\n";
	}

	output << "points\n";
	output << (local_np + static_cast<int>(data.ghost_points.size())) << "\n";
	output << std::setprecision(std::numeric_limits<double>::max_digits10);
	for (int pid = 1; pid <= local_np; ++pid)
	{
		const PointCoordRecord &point = data.local_points_cache.at(static_cast<std::size_t>(pid));
		output << point.xyz[0] << " " << point.xyz[1] << " " << point.xyz[2] << "\n";
	}
	for (const PointCoordRecord &point : data.ghost_points)
	{
		output << point.xyz[0] << " " << point.xyz[1] << " " << point.xyz[2] << "\n";
	}

	output << "endmesh\n";
}

bool WriteVolFromStreams(const std::string &point_table_path,
	const std::string &surface_file,
	const std::string &tet_file,
	const std::string &out_vol_path)
{
	std::ifstream surface_input(surface_file, std::ios::binary);
	if (!surface_input)
	{
		return false;
	}
	std::ifstream tet_input(tet_file, std::ios::binary);
	if (!tet_input)
	{
		return false;
	}
	std::ifstream point_input(point_table_path, std::ios::binary);
	if (!point_input)
	{
		return false;
	}

	EnsureParentDirectory(out_vol_path);
	std::ofstream output(out_vol_path, std::ios::trunc);
	if (!output)
	{
		return false;
	}

	std::vector<FaceRecord> face_buffer;
	std::size_t surface_count = 0;
	int max_geoboundary = 0;
	while (ReadFaceBatch(surface_input, face_buffer, kDefaultStreamBatch) > 0)
	{
		surface_count += face_buffer.size();
		for (const FaceRecord &record : face_buffer)
		{
			max_geoboundary = std::max(max_geoboundary, std::max(record.geoboundary, 1));
		}
	}
	surface_input.clear();
	surface_input.seekg(0, std::ios::beg);

	std::vector<TetRecord> tet_buffer;
	std::size_t tet_count = 0;
	while (ReadTetBatch(tet_input, tet_buffer, kDefaultStreamBatch) > 0)
	{
		tet_count += tet_buffer.size();
	}
	tet_input.clear();
	tet_input.seekg(0, std::ios::beg);

	const int point_count = GetPointRecordCount(point_table_path);

	output << "# Generated by NETGEN stream writer\n\n";
	output << "mesh3d\n";
	output << "dimension\n3\n";
	output << "geomtype\n0\n\n";
	output << "# surfnr\tdomin\tdomout\ttlosurf\tbcprop\n";
	output << "facedescriptors\n";
	output << max_geoboundary << "\n";
	for (int descriptor = 1; descriptor <= max_geoboundary; ++descriptor)
	{
		output << descriptor << " 1 0 " << descriptor << " " << descriptor << "\n";
	}

	output << "surfaceelements\n";
	output << surface_count << "\n";
	while (ReadFaceBatch(surface_input, face_buffer, kDefaultStreamBatch) > 0)
	{
		for (const FaceRecord &record : face_buffer)
		{
			const int descriptor = std::max(record.geoboundary, 1);
			output << " " << descriptor << " 1 1 0 3 "
				<< record.lsvrtx[0] << " "
				<< record.lsvrtx[1] << " "
				<< record.lsvrtx[2] << "\n";
		}
	}

	output << "volumeelements\n";
	output << tet_count << "\n";
	while (ReadTetBatch(tet_input, tet_buffer, kDefaultStreamBatch) > 0)
	{
		for (const TetRecord &record : tet_buffer)
		{
			output << record.domidx << " 4 "
				<< record.vids[0] << " "
				<< record.vids[1] << " "
				<< record.vids[2] << " "
				<< record.vids[3] << "\n";
		}
	}

	output << "points\n";
	output << point_count << "\n";
	output << std::setprecision(std::numeric_limits<double>::max_digits10);
	std::vector<PointCoordRecord> point_buffer(kDefaultStreamBatch);
	while (true)
	{
		point_input.read(reinterpret_cast<char *>(point_buffer.data()),
			static_cast<std::streamsize>(point_buffer.size() * sizeof(PointCoordRecord)));
		const std::streamsize bytes_read = point_input.gcount();
		if (bytes_read == 0)
		{
			break;
		}
		if (bytes_read < 0 || bytes_read % static_cast<std::streamsize>(sizeof(PointCoordRecord)) != 0)
		{
			FailStreamOperation("point table file is truncated while writing .vol: " + point_table_path);
		}
		const std::size_t count = static_cast<std::size_t>(bytes_read / static_cast<std::streamsize>(sizeof(PointCoordRecord)));
		for (std::size_t i = 0; i < count; ++i)
		{
			output << point_buffer[i].xyz[0] << " "
				<< point_buffer[i].xyz[1] << " "
				<< point_buffer[i].xyz[2] << "\n";
		}
	}

	output << "endmesh\n";
	return static_cast<bool>(output);
}

nglib::Ng_Mesh *ClearVolumeElementsOrEquivalent(nglib::Ng_Mesh *mesh)
{
	// nglib 这套接口在当前工程里没有直接暴露“清空 volume elements”的 C API。
	// 因此这里采用兼容方案：重建一个只保留 face descriptor / points / surface elements 的新 mesh，
	// 作为最终 tet replay 的目标容器。
	//
	// 这样可以避免中间轮 refined tetra 长期挂在旧 submesh 里常驻内存。
	// nglib.h does not expose a ClearVolumeElements C wrapper in this tree.
	// Rebuild the mesh with the same descriptors, points, and current surface elements,
	// but without replaying any volume elements.
	nglib::Ng_Mesh *new_mesh = nglib::Ng_NewMesh();

	const int face_num = nglib::Ng_GetNFD(mesh);
	for (int i = 1; i <= face_num; ++i)
	{
		int descriptor[4] = {0};
		nglib::My_Ng_GetFaceDescriptor(mesh, i, descriptor);
		nglib::My_Ng_AddFaceDescriptor(new_mesh, descriptor[0], descriptor[1], descriptor[2], descriptor[3]);
	}

	const int np = nglib::Ng_GetNP(mesh);
	for (int i = 1; i <= np; ++i)
	{
		double xyz[3] = {0.0, 0.0, 0.0};
		int new_idx = 0;
		nglib::Ng_GetPoint(mesh, i, xyz);
		nglib::Ng_AddPoint(new_mesh, xyz, new_idx);
		if (new_idx != i)
		{
			FailStreamOperation("point renumbering changed while clearing volume elements");
		}
	}

	const int nse = nglib::Ng_GetNSE(mesh);
	for (int i = 1; i <= nse; ++i)
	{
		int surfpoints[8] = {0};
		int surfidx = 0;
		const nglib::Ng_Surface_Element_Type et = nglib::Ng_GetSurfaceElement(mesh, i, surfpoints, surfidx);
		nglib::Ng_AddSurfaceElementwithIndex(new_mesh, et, surfpoints, surfidx);
	}

	nglib::Ng_DeleteMesh(mesh);
	return new_mesh;
}

//数组排序，整个数组元素将按升序排序
void SortInt(int *v)
{
	if (v[0] > v[1])
	{
		v[0] ^= v[1];
		v[1] ^= v[0];
		v[0] ^= v[1];
	}
	if (v[1] > v[2])
	{
		v[1] ^= v[2];
		v[2] ^= v[1];
		v[1] ^= v[2];
	}
	if (v[0] > v[1])
	{
		v[0] ^= v[1];
		v[1] ^= v[0];
		v[0] ^= v[1];
	}
}
void sort3int(
	int *v)
{
	if (v[0] > v[1])
	{
		v[0] ^= v[1];
		v[1] ^= v[0];
		v[0] ^= v[1];
	}
	if (v[1] > v[2])
	{
		v[1] ^= v[2];
		v[2] ^= v[1];
		v[1] ^= v[2];
	}
	if (v[0] > v[1])
	{
		v[0] ^= v[1];
		v[1] ^= v[0];
		v[0] ^= v[1];
	}
}
void sort2int(
	int *v)
{
	if (v[1] > v[0])
	{
		v[0] ^= v[1];
		v[1] ^= v[0];
		v[0] ^= v[1];
	}
}


//根据给定的主网格mesh，创建一个子网格submesh
bool NewSubmesh(void *mesh, void *submesh)
{
	//获取主网格mesh的面数
	int faceNum = nglib::Ng_GetNFD((nglib::Ng_Mesh *)mesh);
	for (int i = 2; i < faceNum + 1; i++)
	{
		int x[4];
		//获取面的描述符，并将其存储在数组x中
		nglib::My_Ng_GetFaceDescriptor((nglib::Ng_Mesh *)mesh, i, x);
		//将数组x中的描述符添加到子网格submesh中
		nglib::My_Ng_AddFaceDescriptor((nglib::Ng_Mesh *)submesh, x[0], x[1], x[2], x[3]);
	}
	return true;
}

//使用了METIS库进行网格划分的函数
idx_t *PartitionMetis(void *mesh, int numParts, idx_t *eptr, idx_t *eind, idx_t *edest, idx_t *ndest)
{
	idx_t ne; // 网格中的元素数目
	idx_t nn; // 网格中的节点数目
	idx_t ncommon = 3;	//两个相邻元素之间的共享节点数目
	idx_t objval;	//划分结果的优化目标值
	int vrts[4];	//存储元素的节点索引数组
	int rc; 		// METIS库的返回代码
	int i;			//
	int elemno;		//元素编号
	int domainidx;	//元素所属的域索引
	idx_t options[METIS_NOPTIONS];//METIS库的选项数组

	METIS_SetDefaultOptions(options);	//设置默认的 METIS 选项
	options[METIS_OPTION_CONTIG] = 0;
	nglib::Ng_Mesh *ngMesh = (nglib::Ng_Mesh *)mesh;
	nn = nglib::Ng_GetNP(ngMesh);
	ne = nglib::Ng_GetNE(ngMesh);
	// printf("MY: (nn,ne) (%d 9%d)\n", nn, ne);
	MYMALLOC(eptr, idx_t *, ((ne + 1) * sizeof(idx_t))); // memory allocation
	MYMALLOC(edest, idx_t *, (ne * sizeof(idx_t)));
	MYMALLOC(eind, idx_t *, (4 * ne * sizeof(idx_t)));
	MYMALLOC(ndest, idx_t *, (nn * sizeof(idx_t)));
	eptr[0] = 0;
	i = 0;

	//获取元素的节点索引，并将其存储在 eind 数组中。
	for (elemno = 1; elemno <= ne; elemno++)
	{
		nglib::Ng_GetVolumeElement(ngMesh, elemno, vrts, domainidx);
		eind[i++] = vrts[0] - 1;
		eind[i++] = vrts[1] - 1;
		eind[i++] = vrts[2] - 1;
		eind[i++] = vrts[3] - 1;
		eptr[elemno] = eptr[elemno - 1] + 4;
	}
	// partition function for the metis
	//  printf("NE = %d\n", ne);
	//  printf("NN = %d\n", nn);
	//  printf("numParts = %d\n", numParts);

	//调用 METIS_PartMeshDual 函数进行网格划分
	rc = METIS_PartMeshDual(&ne, &nn, eptr, eind, NULL, NULL, &ncommon, &numParts, NULL, options,
							&objval, edest, ndest);
	// printf("MY: metis objval = 9%d\n",objval);
	// printf("Partipartition succeeded!");
	free(eptr);
	free(eind);
	free(ndest);
	//返回 edest 数组，其中包含了元素的目标划分信息
	return edest;
}

//对给定的网格 mesh 进行划分，并返回划分后的结果。
idx_t *PartitionMesh(void *mesh, int parts)
{
	// 用于存储划分过程中的中间结果
	idx_t *eptr = NULL;
	idx_t *eind = NULL;
	idx_t *edest = NULL;
	idx_t *ndest = NULL;

	//该函数会使用 METIS 库对网格进行划分，并返回划分结果到 edest 数组中
	edest = PartitionMetis((nglib::Ng_Mesh *)mesh, parts, eptr, eind, edest, ndest);
	// Extract surface mesh and refine
	// ExtractPartit ionSurfaceMesh ((nglib::Ng_ Mesh*)mesh, numlevels, edest, maxbarycoord, surfMap) ;

	//函数返回了 edest 数组，即划分后的结果
	return edest;
}

//从划分后的网格中提取表面网格，并将结果存储在 facemap 中
void ExtractPartitionSurfaceMesh(void *mesh, idx_t *edest, std::map<int, xdMeshFaceInfo> &facemap)
{

	//传入的 mesh 转换为 nglib::Ng_Mesh 类型的指针 newMesh
	nglib::Ng_Mesh *newMesh = (nglib::Ng_Mesh *)mesh;
	int fids[4]; // four indexes of the tetrahedron face
	int vrts[4];
	// index of the four vertices of the tetrahedra
	int orient[4];
	int p, i, k, elemno;
	int domainidx;
	bool netgenmeshupdate = true;
	
	//获取当前进程的 MPI 通信相关信息，包括进程编号 id 和进程总数 size。
	int id, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//获取网格中的表面元素数量和体元素数量
	int nse = nglib::Ng_GetNSE(newMesh);
	int ne = nglib::Ng_GetNE(newMesh);
	surfMap_t surfMap;
	GetSurfPoints(newMesh, surfMap);
	time_t start_time,time1,time2,time3,time4;
	if(id == 0) {
		start_time = time(NULL);
	}
	//每个进程计算了划分后负责处理的元素数量。
	std::vector<int> every_sub_ne(size);
	std::vector<int> every_offset(size);
	int sub_ne = ne / size;
	for(int i = 0; i < size; i++) {
		every_sub_ne[i] = sub_ne*4;
		every_offset[i] = sub_ne*4 * i;
	}
	every_sub_ne[size-1] = (ne - (sub_ne * (size-1)))*4;
	int start_ne = sub_ne * id + 1;
	int end_ne = start_ne + sub_ne-1;
	if(id == size - 1) {
		end_ne = ne;
	}

	// std::vector<fid_xdMeshFaceInfo> sub_face_map;
	sub_ne = end_ne-start_ne+1;
	std::vector<fid_xdMeshFaceInfo> sub_face_map(sub_ne*4);
	int sub_face_map_index = 0;

	//函数根据每个进程负责的元素范围，遍历划分后的网格的元素
	for (elemno = start_ne; elemno <= end_ne; elemno++)
	{
		p = edest[elemno - 1];
		// the edest storage area decomposes after each body element belongs to the partition
		// if (netgenmeshupdate) printf("MY: starting update\n")

		//获取其面的索引
		nglib::My_Ng_GetElement_Faces(newMesh, elemno, fids, orient, netgenmeshupdate);
		// get the four surface indexes for each body elemetn

		//获取顶点的索引
		netgenmeshupdate = false;
		nglib::Ng_GetVolumeElement(newMesh, elemno, vrts, domainidx);


		// get the four vertex indexes for each body element
		for (k = 0; k < 4; k++)
		{
			//调用ExtractSurfaceMesh函数提取表面网格，将面信息存储在sub_face_map中
			ExtractSurfaceMesh(newMesh, fids[k], p, vrts, sub_face_map.data(), sub_face_map_index, domainidx);
			sub_face_map_index++;
		}

	}

	//使用Allgather_Face_Map函数收集所有进程的结果，得到最终的表面网格
	Allgather_Face_Map(facemap, sub_face_map.data(), ne, sub_ne, every_sub_ne.data(), every_offset.data());
	//函数在进程编号为 0 的进程输出总运行时间
	if(id == 0) {
		time_t end_time = time(NULL);
		std::cout << "id0 sum runtime is :" << end_time - start_time << "s" << std::endl;
	}
}

void GetSurfPoints(void *mesh, surfMap_t &surfMap)
{
	nglib::Ng_Mesh *newMesh = (nglib::Ng_Mesh *)mesh;
	int surfidx;
	//获取网格中的表面元素的数量
	int nse = nglib::Ng_GetNSE(newMesh);
	//存储每个表面元素的节点索引
	int *surfpoints = new int[3];
	for (int i = 0; i < nse; i++)
	{
		//获取每个表面元素的节点索引和表面索引
		nglib::Ng_GetSurfaceElement(newMesh, i + 1, surfpoints, surfidx);
		FacePoints fp;
		//将节点索引排序后，将排序后的节点索引作为键，表面索引作为值，存储在 surfMap 中
		std::sort(surfpoints, surfpoints + 3);
		fp.p[0] = surfpoints[0];
		fp.p[1] = surfpoints[1];
		fp.p[2] = surfpoints[2];
		// surfMap[fp] = nglib::GetBoundaryID(newMesh, i + 1);
		surfMap[fp] = surfidx;
	}
}

//提取网格中的表面网格信息，并将结果存储在 sub_face_map 中。
void ExtractSurfaceMesh(void *mesh, int fid, int procid, int *tverts, fid_xdMeshFaceInfo *sub_face_map, int sub_face_map_index, int domainidx)
{
	nglib::Ng_Mesh *newMesh = (nglib::Ng_Mesh *)mesh;
	xdMeshFaceInfo finfo;
	std::map<int, xdMeshFaceInfo>::iterator it;
	int fverts[3];
	int outw;
	//通过传入的表面元素的索引 fid，获取该表面元素的顶点索引 fverts
	nglib::My_Ng_GetFace_Vertices(newMesh, fid, fverts); // get an index of the surface points according to the id of the surface
	// it = facemap.find(fid);

	//根据传入的顶点索引 tverts 和表面顶点索引 fverts，计算表面的外向法向量
	outw = ExtractFaceOutward(tverts, fverts);

	//将表面的外向法向量、进程编号、顶点索引和域索引存储在 finfo 中
	finfo.outw = outw;
	finfo.procids[0] = procid;
	finfo.procids[1] = -1;
	finfo.svrtx[0][0] = fverts[0];
	if (!outw)
	{
		finfo.svrtx[0][1] = fverts[1];
		finfo.svrtx[0][2] = fverts[2];
	}
	else
	{
		finfo.svrtx[0][1] = fverts[2];
		finfo.svrtx[0][2] = fverts[1];
	}
	finfo.domainidx[0] = domainidx;

	// 将表面元素的索引 fid 和上述的 finfo 存储在 fmfi 中
	fid_xdMeshFaceInfo fmfi;
	fmfi.fid = fid;
	fmfi.mfi = finfo;

	//函数将 fmfi 存储在 sub_face_map 的相应位置，以便后续处理和使用
	sub_face_map[sub_face_map_index] = fmfi;

	// if (it == facemap.end())
	// {
	// 	finfo.outw = outw;
	// 	finfo.procids[0] = procid;
	// 	finfo.procids[1] = -1;
	// 	finfo.svrtx[0][0] = fverts[0];
	// 	if (!outw)
	// 	{
	// 		finfo.svrtx[0][1] = fverts[1];
	// 		finfo.svrtx[0][2] = fverts[2];
	// 	}
	// 	else
	// 	{
	// 		finfo.svrtx[0][1] = fverts[2];
	// 		finfo.svrtx[0][2] = fverts[1];
	// 	}
	// 	finfo.domainidx[0] = domainidx;
	// 	facemap[fid] = finfo;
	// }
	// else if ((it->second).procids[0] == procid)
	// { // internal face
	// 	facemap.erase(it);
	// }
	// else
	// { // partition boundary face
	// 	(it->second).procids[1] = procid;
	// 	(it->second).svrtx[1][0] = fverts[0];
	// 	if (!outw)
	// 	{
	// 		(it->second).svrtx[1][1] = fverts[1];
	// 		(it->second).svrtx[1][2] = fverts[2];
	// 	}
	// 	else
	// 	{
	// 		(it->second).svrtx[1][1] = fverts[2];
	// 		(it->second).svrtx[1][2] = fverts[1];
	// 	}
	// 	(it->second).domainidx[1] = domainidx;
	// }
}


//自定义MPI类型
MPI_Datatype fid_xdMeshFaceInfo::MPI_type;
//构建MPI类型

//定义了一个名为 fid_xdMeshFaceInfo 的结构体
//在结构体中添加了一个函数 build_MPIType，用于构建 MPI 自定义数据类型
void fid_xdMeshFaceInfo::build_MPIType() {
	//用于存储每个成员变量的长度
	int block_lengths[5];
	MPI_Aint displacements[5];
	MPI_Aint addresses[5], add_start;
	//用于存储每个成员变量的 MPI 数据类型。
	MPI_Datatype typelist[5];

	//用于获取每个成员变量的地址。
	fid_xdMeshFaceInfo temp;

	typelist[0] = MPI_INT;
	block_lengths[0] = 1;
	// MPI_Get_address 函数，获取了 temp 对象中每个成员变量的地址，并将这些地址存储在 addresses 数组中。
	MPI_Get_address(&temp.fid, &addresses[0]);

	typelist[1] = MPI_INT;
	block_lengths[1] = 2;
	MPI_Get_address(&temp.mfi.procids, &addresses[1]);

	typelist[2] = MPI_INT;
	block_lengths[2] = 6;
	MPI_Get_address(&temp.mfi.svrtx, &addresses[2]);

	typelist[3] = MPI_INT;
	block_lengths[3] = 2;
	MPI_Get_address(&temp.mfi.domainidx, &addresses[3]);

	typelist[4] = MPI_SHORT;
	block_lengths[4] = 1;
	MPI_Get_address(&temp.mfi.outw, &addresses[4]);

	//通过计算每个成员变量相对于 temp 对象起始地址的偏移量，将这些偏移量存储在 displacements 数组中。
	MPI_Get_address(&temp, &add_start);
	for(int i = 0; i < 5; i++) displacements[i] = addresses[i] - add_start;
	//基于上述定义的成员变量数据类型、长度和偏移量，创建了一个新的 MPI 自定义数据类型 MPI_type
	MPI_Type_create_struct(5, block_lengths, displacements, typelist, &MPI_type);
	//
	MPI_Type_commit(&MPI_type);
}


//	-func  : 将提取的子面网格（face_map）合并到std::map中
//	-param : facemap 输出的std::map 面网格
//  -param : sub_face_map 输入的子网格


void Allgather_Face_Map(std::map<int, xdMeshFaceInfo> &facemap, fid_xdMeshFaceInfo *sub_face_map, int ne, int sub_ne, int *every_sub_ne, int *every_offset) {
	int rank;
	//获取当前进程的MPI等级
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//进行数据的同步
	MPI_Barrier(MPI_COMM_WORLD);
	//创建自定义的MPI数据类型
	fid_xdMeshFaceInfo::build_MPIType();
	// fid_xdMeshFaceInfo *sub_face_maps = new fid_xdMeshFaceInfo[ne*4];
	//分配内存空间，创建用于存储合并后面映射数据的数组
	fid_xdMeshFaceInfo *sub_face_maps = (fid_xdMeshFaceInfo*)malloc(sizeof(fid_xdMeshFaceInfo) *(ne)*4);

	
	/*将每个核所求得的sub_face_map全合并*/
	//将子进程的面映射数据进行全局合并
	MPI_Allgatherv(sub_ne > 0 ? sub_face_map : nullptr, sub_ne*4, fid_xdMeshFaceInfo::MPI_type,
	 sub_face_maps, every_sub_ne, every_offset, fid_xdMeshFaceInfo::MPI_type,
	 MPI_COMM_WORLD);
	//释放自定义的数据类型
	MPI_Type_free(&fid_xdMeshFaceInfo::MPI_type);
	//遍历合并后的面映射数据数组，根据面ID更新facemap
	for(int i = 0; i < ne*4; i++) {
		int cur_fid = sub_face_maps[i].fid;
		auto it = facemap.find(cur_fid);
		if(it == facemap.end()) {
			facemap[cur_fid] = sub_face_maps[i].mfi;
		} else if((it->second).procids[0] == sub_face_maps[i].mfi.procids[0]) {
			facemap.erase(it);
		} else {
			(it->second).procids[1] = sub_face_maps[i].mfi.procids[0];
			(it->second).svrtx[1][0] = sub_face_maps[i].mfi.svrtx[0][0];
			// if(!sub_face_maps[i].mfi.outw) {
				(it->second).svrtx[1][1] = sub_face_maps[i].mfi.svrtx[0][1];
				(it->second).svrtx[1][2] = sub_face_maps[i].mfi.svrtx[0][2];
			// } else {
			// 	(it->second).svrtx[1][1] = sub_face_maps[i].mfi.svrtx[0][2];
			// 	(it->second).svrtx[1][2] = sub_face_maps[i].mfi.svrtx[0][1];
			// }
			(it->second).domainidx[1] = sub_face_maps[i].mfi.domainidx[0];
		}
	}

	//释放内存空间
	free(sub_face_maps);
}


//该函数主要用于确定面的朝向，根据输入的顶点数组和面顶点数组，
//通过匹配顶点的方式来判断面是否为外向面。如果面为外向面，返回1；否则，返回0。
int ExtractFaceOutward(int *tverts, int *fverts)
{
	int i, j, s, t;
	int sumt;
	int tetraor[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
	// int tetraor[4][3] = { {1,3,2}, {0,2,3}, {0,3,1}, {0,1,2} };
	//使用循环遍历两个顶点数组，找到共享的顶点。
	for (i = 0; i < 2; i++)
	{
		for (s = 0; s < 3; s++)
		{
			if (tverts[i] == fverts[s])
				break;
		}
		if (s < 3)
			break;
	}
	//如果没有找到共享的顶点，则输出错误信息并退出。
	if (s == 3)
	{
		printf("MY: error in outward face computation (1). \n");
		exit(1);
	}

	//使用两层循环遍历四个可能的面方向。
	sumt = 0;
	for (int i = 0; i < 4; i++)
	{
		t = 0;
		for (int j = 0; j < 3; j++)
		{
			//检查顶点数组中的顶点是否与面顶点数组中的顶点相匹配。
			if (tverts[tetraor[i][j]] == fverts[(s + j) % 3])
				t++;
		}
		if (t == 3)
			sumt++;
	}
	//如果匹配的顶点数为3，则表明该面是外向面
	if (sumt == 1)
	{
		return (1);	//如果找到外向面，则返回1。
	} //如果匹配的顶点数大于3，则输出错误信息并退出。
	else if (sumt > 1) //如果存在多个外向面或计算错误，则输出错误信息并退出
	{
		printf("MY: error in outward face computation (2).\n");
		exit(1);
	}
	return (0);
}


//根据三角面的顶点索引在表面映射中查找对应的表面元素
int FindSurfElem(void *mesh, int *elem, surfMap_t &surfMap)
{
	nglib::Ng_Mesh *newMesh = (nglib::Ng_Mesh *)mesh;
	int nse = nglib::Ng_GetNSE(newMesh);
	//将输入的三角面顶点索引排序，以确保索引数据的一致
	std::sort(elem, elem + 3);
	//创建一个包含排序后顶点索引的FacePoints对象。
	FacePoints fp;
	fp.p[0] = elem[0];
	fp.p[1] = elem[1];
	fp.p[2] = elem[2];
	//在表面映射数据结构中查找具有相同顶点索引的键值对
	surfMap_t::const_iterator it = surfMap.find(fp);
	if (it == surfMap.end())
		return -1;		//返回-1表示未找到对应的表面元素。
	else
		return it->second; //如果找到匹配的键值对，返回对应的表面元素。
}

//在网格中创建面元素
void PartFaceCreate(void *mesh, int belongNumberPartition, std::map<int, xdMeshFaceInfo> &facemap,
					int maxbarycoord, void *submesh, std::map<int, int> &g2lvrtxmap, std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::list<xdFace> &newfaces)
{
	// 根据 facemap 为当前分区构造初始表面三角形 newfaces。
	//
	// 这一步仍然保留原工程的“submesh 是唯一真实网格对象”的原则：
	// - 顶点直接加到 submesh；
	// - newfaces 只保存待细分/待回灌的表面工作集；
	// - 同时建立 barycentric 的双向映射，为后续 stream dump/replay 做准备。
	nglib::Ng_Mesh *newMesh = (nglib::Ng_Mesh *)mesh;
	std::map<int, xdMeshFaceInfo>::iterator itf;
	std::map<int, int>::iterator itv;
	int fid;
	xdMeshFaceInfo finfo;
	xdFace f{};
	int lsvrtx[3];
	int pid;
	double xyz[3];
	int idx = -1;
	std::map<IntPair, int, IntPairCompare> facepairs;
	int faceidx;
	int faceNum;
	Barycentric barycv;
	IntPair fpr;
	std::map<IntPair, int, IntPairCompare>::iterator fi;
	surfMap_t surfMap;
	//获取原始网格中的表面点信息，将其存储在表面映射数据结构surfMap中。
	GetSurfPoints(newMesh, surfMap);
	// add vertices to the mesh
	//遍历facemap中的每个面元素，处理属于当前分区的面元素。
	for (itf = facemap.begin(); itf != facemap.end(); ++itf)
	{
		fid = itf->first;
		finfo = itf->second;
		// insert face vertices into verts map
		for (int j = 0; j < 2; j++)
		{
			pid = finfo.procids[j];
			if (pid == belongNumberPartition)
			{
				// Ng_ AddSurfaceBlement (submesh, et, finfo. svrtx[j]);
				for (int k = 0; k < 3; k++)
				{
					itv = g2lvrtxmap.find(finfo.svrtx[j][k]);
					// finfo. svrtx[j][k] the index of the vertex in which zone
					if (itv == g2lvrtxmap.end())
					{
						// the vertex is not stored in the g2lvrtxmap, then put in
						nglib::Ng_GetPoint((nglib::Ng_Mesh *)mesh, finfo.svrtx[j][k], xyz); // netgen obtains the coordinates of the point based on the point index
						nglib::Ng_AddPoint((nglib::Ng_Mesh *)submesh, xyz, idx);
						// add the resulting coordinates to the new grid
						g2lvrtxmap[finfo.svrtx[j][k]] = idx;
						barycv = InitBarycv(finfo.svrtx[j][k], maxbarycoord);
						RegisterBarycentricVertex(baryc2locvrtxmap, locvrtx2barycmap, barycv, idx);
					}
				}
			}
		}
	}
	int Isbound;
	// add face element to the grid
	//再次处理属于当前分区的面元素。
	for (itf = facemap.begin(); itf != facemap.end(); ++itf)
	{
		fid = itf->first;
		finfo = itf->second;
		for (int j = 0; j < 2; j++)
		{
			if (j == 1 && finfo.procids[1] == finfo.procids[0])
				// if (finfo.procids[1] == finfo.procids[0])
				break;
			pid = finfo.procids[j]; // get the partition where the face element belongs
			if (pid == belongNumberPartition)
			{
				f.outw = finfo.outw;
				Isbound = (finfo.procids[1] == -1) ? 1 : 0;	 // determine whether it is an interface, not the value of interface geoboundary is set to 1,and the value of interface
				f.patbound = (!Isbound && finfo.procids[0] != finfo.procids[1]) ? 1 : 0;
				for (int k = 0; k < 3; ++k)
				{
					const int gv = finfo.svrtx[j][k];
					itv = g2lvrtxmap.find(gv);
					if (itv == g2lvrtxmap.end())
					{
						nglib::Ng_GetPoint((nglib::Ng_Mesh *)mesh, gv, xyz);
						nglib::Ng_AddPoint((nglib::Ng_Mesh *)submesh, xyz, idx);
						g2lvrtxmap[gv] = idx;
						barycv = InitBarycv(gv, maxbarycoord);
						RegisterBarycentricVertex(baryc2locvrtxmap, locvrtx2barycmap, barycv, idx);
						f.lsvrtx[k] = idx;
					}
					else
					{
						f.lsvrtx[k] = itv->second;
					}
					f.barycv[k] = InitBarycv(gv, maxbarycoord);
				}
				f.geoboundary = FindSurfElem((nglib::Ng_Mesh *)mesh, finfo.svrtx[j], surfMap);

				fpr.x = finfo.domainidx[j];
				fpr.y = finfo.domainidx[(j + 1) % 2];
				if ((f.geoboundary == -1 || finfo.procids[0] != finfo.procids[1]) && !Isbound)
				{
					fi = facepairs.find(fpr);
					if (fi == facepairs.end())
					{
						faceNum = nglib::Ng_GetNFD((nglib::Ng_Mesh *)submesh);
						// faceidx = nglib::My_Ng_AddFaceDescriptor((nglib::Ng_Mesh*)submesh, faceNum, finfo.domainidx[j], 0, 0);
						faceidx = nglib::My_Ng_AddFaceDescriptor((nglib::Ng_Mesh *)submesh, faceNum + 1, finfo.domainidx[j], 0, 0);
						f.geoboundary = faceidx;
						facepairs[fpr] = faceidx;
					}
					else
					{
						f.geoboundary = facepairs[fpr];
					}
				}
				// std::cout << f.geoboundary << std::endl;
				newfaces.push_back(f);
				// place the face element to the newfaces
			}
		}
	}
}

void DumpFacesToFileAndClear(std::list<xdFace> &newfaces, const std::string &filepath)
{
	// 将初始/中间 newfaces 立刻落成二进制 face stream，然后释放 list 节点。
	// 这是切断“newfaces 长时间常驻内存”的第一步。
	EnsureParentDirectory(filepath);
	std::ofstream output(filepath, std::ios::binary | std::ios::trunc);
	if (!output)
	{
		FailStreamOperation("failed to open face stream output file: " + filepath);
	}

	std::vector<FaceRecord> buffer;
	buffer.reserve(kDefaultStreamBatch);
	for (const xdFace &face : newfaces)
	{
		buffer.push_back(ToFaceRecord(face));
		if (buffer.size() >= kDefaultStreamBatch)
		{
			FlushFaceBuffer(output, buffer);
		}
	}
	FlushFaceBuffer(output, buffer);

	std::list<xdFace>().swap(newfaces);
}

void DumpNewFacesToSurfaceStream(
	const std::list<xdFace> &newfaces,
	const std::string &filepath)
{
	EnsureParentDirectory(filepath);

	std::ofstream output(filepath, std::ios::binary | std::ios::trunc);
	if (!output)
	{
		FailStreamOperation("failed to open initial surface stream file from newfaces: " + filepath);
	}

	std::vector<FaceRecord> buffer;
	buffer.reserve(kDefaultStreamBatch);

	for (const xdFace &face : newfaces)
	{
		FaceRecord record = ToFaceRecord(face);
		buffer.push_back(record);
		if (buffer.size() >= kDefaultStreamBatch)
		{
			FlushFaceBuffer(output, buffer);
		}
	}

	FlushFaceBuffer(output, buffer);
}

void RefineOneLevelToFile(void *submesh,
	const std::string &infile,
	const std::string &outfile,
	std::size_t batch_faces,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	std::map<IntPair, int, IntPairCompare> &edgemap)
{
	// 将一个 surface stream 层级细分成下一层。
	// 这里只做“文件 -> 文件”的一级转换，不直接回灌 submesh 的 surface elements，
	// 便于多级 refine 串起来按层流式处理。
	StreamRefineFacesToFile(submesh, infile, outfile, batch_faces, baryc2locvrtxmap, locvrtx2barycmap, edgemap, nullptr);
}

void AddFacesFromFileToMesh(void *submesh, const std::string &filepath, std::size_t batch_faces)
{
	// 最终层 face stream 回灌到 submesh。
	// 注意这里仍然是边读边加，不会把最终层所有三角面再读回一个巨大的 list。
	std::ifstream input(filepath, std::ios::binary);
	if (!input)
	{
		FailStreamOperation("failed to open face stream input file: " + filepath);
	}

	std::vector<FaceRecord> buffer;
	while (ReadFaceBatch(input, buffer, batch_faces) > 0)
	{
		for (const FaceRecord &record : buffer)
		{
			const xdFace face = FromFaceRecord(record);
			int lsvrtx[3] = {face.lsvrtx[0], face.lsvrtx[1], face.lsvrtx[2]};
			nglib::Ng_AddSurfaceElementwithIndex((nglib::Ng_Mesh *)submesh, nglib::NG_TRIG, lsvrtx, face.geoboundary);
		}
	}
}

void Refine_Stream(void *submesh,
	int numlevels,
	int belongNumberPartition,
	const std::string &stream_dir,
	std::size_t batch_faces,
	bool keep_stream_files,
	std::list<xdFace> &newfaces,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	std::map<IntPair, int, IntPairCompare> &edgemap)
{
	// surface 完整体流式链路：
	// 1. 先把 PartFaceCreate 产生的 L0 newfaces 落盘；
	// 2. 每一层都执行 “faces_Lk.bin -> faces_L(k+1).bin”；
	// 3. 最终层再直接回灌到 submesh。
	//
	// 这样既保留原 Refine() 的几何规则，又避免 list<xdFace> 随层级指数膨胀。
	const std::string rank_prefix = "faces_rank" + std::to_string(belongNumberPartition);
	std::string current_file = stream_dir + "/" + rank_prefix + "_L0.bin";
	DumpFacesToFileAndClear(newfaces, current_file);

	for (int level = 0; level < numlevels; ++level)
	{
		const std::string next_file = stream_dir + "/" + rank_prefix + "_L" + std::to_string(level + 1) + ".bin";
		RefineOneLevelToFile(submesh, current_file, next_file, batch_faces, baryc2locvrtxmap, locvrtx2barycmap, edgemap);
		if (!keep_stream_files)
		{
			std::filesystem::remove(current_file);
		}
		current_file = next_file;
	}

	newfaces.clear();
	std::ifstream input(current_file, std::ios::binary);
	if (!input)
	{
		FailStreamOperation("failed to open face stream input file: " + current_file);
	}
	std::vector<FaceRecord> buffer;
	while (ReadFaceBatch(input, buffer, batch_faces) > 0)
	{
		for (const FaceRecord &record : buffer)
		{
			const xdFace face = FromFaceRecord(record);
			newfaces.push_back(face);
			int lsvrtx[3] = {face.lsvrtx[0], face.lsvrtx[1], face.lsvrtx[2]};
			nglib::Ng_AddSurfaceElementwithIndex((nglib::Ng_Mesh *)submesh, nglib::NG_TRIG, lsvrtx, face.geoboundary);
		}
	}

	if (!keep_stream_files)
	{
		std::filesystem::remove(current_file);
	}
}

void Refine(void *submesh, int numlevels, int belongNumberPartition, std::list<xdFace> &newfaces, std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::map<IntPair, int, IntPairCompare> &edgemap)
{
	// 原版非流式 surface refine 路径保留，用于 A/B 对比。
	// 唯一额外补上的能力是同步维护 locvrtx2barycmap，保证后续可以从 submesh 反查 barycentric。
	(void)belongNumberPartition;
	std::list<xdFace>::iterator li = newfaces.begin();
	std::vector<xdFace> children;
	children.reserve(4);

	for (int l = 0; l < numlevels; ++l)
	{
		const int listsize = static_cast<int>(newfaces.size());
		for (int t = 0; t < listsize; ++t)
		{
			const xdFace face = *li;
			RefineFaceAndCollectChildren((nglib::Ng_Mesh *)submesh, face, children, baryc2locvrtxmap, locvrtx2barycmap, edgemap, nullptr);
			for (const xdFace &child : children)
			{
				newfaces.push_back(child);
			}
			li = newfaces.erase(li);
		}
		li = newfaces.begin();
	}

	for (li = newfaces.begin(); li != newfaces.end(); ++li)
	{
		nglib::Ng_AddSurfaceElementwithIndex((nglib::Ng_Mesh *)submesh, nglib::NG_TRIG, (*li).lsvrtx, (*li).geoboundary);
	}
}


//初始化重心坐标（Barycentric）结构
Barycentric InitBarycv(int v, int maxbarycoord)
{
	Barycentric p;
	p.gvrtx[0] = 0;
	p.gvrtx[1] = 0;
	p.gvrtx[2] = v;
	p.coord[0] = 0;
	p.coord[1] = 0;
	p.coord[2] = maxbarycoord;
	p.newgid = 0;
	return (p);
}

//计算两个重心坐标（Barycentric）之间的中点
void BarycMidPoint(Barycentric p1, Barycentric p2, Barycentric &res)
{	
	//创建一个集合 s，用于存储两个重心坐标中非零坐标对应的顶点索引
	std::set<int> s;
	std::set<int>::iterator is;
	int k;
	short w1, w2;
	//遍历 p1 和 p2 的坐标数组，将非零坐标对应的顶点索引插入集合 s 中。
	for (int i = 0; i < 3; i++)
	{
		if (p1.coord[i])
			s.insert(p1.gvrtx[i]);
		if (p2.coord[i])
			s.insert(p2.gvrtx[i]);
	}
	//检查集合 s 的大小，如果超过了 3，说明出现了错误，打印错误信息。
	if (s.size() > 3)
	{
		printf("Error in barycentric coordinates\n");
	}
	// the global result of the center of mass of the third point from the index of teo pl,p2 points (the third point is the insertion point)
	k = 0;
	for (is = s.begin(); is != s.end(); is++)
	{
		res.gvrtx[k] = *is;
		k++;
	}
	//如果集合 s 中的顶点索引不足三个，将剩余的索引用零填充。
	while (k < 3)
	{
		res.gvrtx[k] = 0;
		k++;
	}
	//对 res 的顶点索引数组进行排序。
	SortInt(res.gvrtx);
	// to ate the index of the insertion point
	//针对每个顶点索引，从 p1 和 p2 的坐标数组中找到对应的权重值 w1 和 w2。
	for (int i = 0; i < 3; i++)
	{
	w1 = w2 = 0;
		for (int j = 0; j < 3; j++)
		{
			if (res.gvrtx[i] == p1.gvrtx[j])
			{
				w1 = p1.coord[j];
				break;
			}
		}
		for (int j = 0; j < 3; j++)
		{
			if (res.gvrtx[i] == p2.gvrtx[j])
			{
				w2 = p2.coord[j];
				break;
			}
		}
		// determine whether the index of the pl,p2 two points has the same as the index of the insertion point,and then get the coordinates of the insertion point
		// based on the coordinates of the same index as the pl,p2 two points
		//根据 w1 和 w2 的值，计算插入点的权重值，即取 w1 和 w2 的平均值，并赋值给结果重心坐标 res 的坐标数组 coord。
		res.coord[i] = (w1 + w2) / 2;
	}
	res.newgid = 0;
}

void DumpCurrentSurfaceElementsToFile(
	void *submesh,
	const std::string &filepath,
	const std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	const std::map<int, Barycentric> &locvrtx2barycmap,
	const std::set<FaceKey> *saved_patbound_faces)
{
	// 从“当前 submesh 的真实 surface elements”反向重建 surface stream。
	//
	// 这是完整体 stream 方案的关键桥梁：
	// volume refine 不能再依赖旧内存中的 newfaces，
	// 所以必须能从当前 mesh 重新导出 boundary face 及其 barycentric 信息。
	EnsureParentDirectory(filepath);
	std::ofstream output(filepath, std::ios::binary | std::ios::trunc);
	if (!output)
	{
		FailStreamOperation("failed to open surface dump file: " + filepath);
	}

	// locvrtx2barycmap 是反向查找表，使 stream 路径可以直接由本地点号反查 barycentric，
	// 不需要全表扫描 baryc2locvrtxmap。
	std::vector<FaceRecord> buffer;
	buffer.reserve(kDefaultStreamBatch);
	nglib::Ng_Mesh *mesh = (nglib::Ng_Mesh *)submesh;
	const int nse = nglib::Ng_GetNSE(mesh);
	for (int i = 0; i < nse; ++i)
	{
		int surfpoints[3];
		int surfidx = 0;
		nglib::Ng_GetSurfaceElement(mesh, i + 1, surfpoints, surfidx);

		xdFace face{};
		face.outw = 0;
		const FaceKey fk = make_face_key(surfpoints[0], surfpoints[1], surfpoints[2]);
		if (saved_patbound_faces)
			face.patbound = saved_patbound_faces->count(fk) ? 1 : 0;
		else
			face.patbound = 0;
		face.geoboundary = surfidx;
		for (int k = 0; k < 3; ++k)
		{
			face.lsvrtx[k] = surfpoints[k];
			auto bary_it = locvrtx2barycmap.find(surfpoints[k]);
			if (bary_it == locvrtx2barycmap.end())
			{
				FailStreamOperation("missing barycentric reverse mapping for surface vertex " + std::to_string(surfpoints[k]));
			}
			face.barycv[k] = bary_it->second;
			face.barycv[k].newgid = 0;
			auto forward_it = baryc2locvrtxmap.find(face.barycv[k]);
			if (forward_it == baryc2locvrtxmap.end() || forward_it->second != surfpoints[k])
			{
				FailStreamOperation("surface vertex barycentric mapping is inconsistent for vertex " + std::to_string(surfpoints[k]));
			}
		}
		buffer.push_back(ToFaceRecord(face));
		if (buffer.size() >= kDefaultStreamBatch)
		{
			FlushFaceBuffer(output, buffer);
		}
	}
	FlushFaceBuffer(output, buffer);
}

void DumpCurrentVolumeElementsToFile(void *submesh, const std::string &filepath)
{
	// 将当前 submesh 中的所有四面体导出为 tet stream。
	// 这份文件是 volume stream 每一轮 refine 的输入工作集。
	EnsureParentDirectory(filepath);
	std::ofstream output(filepath, std::ios::binary | std::ios::trunc);
	if (!output)
	{
		FailStreamOperation("failed to open tet dump file: " + filepath);
	}

	std::vector<TetRecord> buffer;
	buffer.reserve(kDefaultStreamBatch);
	nglib::Ng_Mesh *mesh = (nglib::Ng_Mesh *)submesh;
	const int ne = nglib::Ng_GetNE(mesh);
	for (int i = 0; i < ne; ++i)
	{
		int vids[4];
		int domidx = 0;
		nglib::Ng_GetVolumeElement(mesh, i + 1, vids, domidx);
		buffer.push_back(ToTetRecord(vids, domidx));
		if (buffer.size() >= kDefaultStreamBatch)
		{
			FlushTetBuffer(output, buffer);
		}
	}
	FlushTetBuffer(output, buffer);
}

void Refineforvol_Stream(void *submesh,
	const std::string &surface_infile,
	const std::string &tet_infile,
	const std::string &surface_outfile,
	const std::string &tet_outfile,
	const std::string &point_table_path,
	const std::string &output_path,
	std::size_t batch_faces,
	std::size_t batch_tets,
	int &next_point_id,
	std::size_t edge_shards,
	int round,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, Barycentric> &locvrtx2barycmap,
	std::map<IntPair, int, IntPairCompare> &edgemap)
{
	// 完整体 volume stream 路径。
	//
	// 输入不再是内存里的 newfaces，而是：
	// - 当前轮的 boundary surface stream
	// - 当前轮的 tet stream
	// - 当前 submesh 中已经存在的点
	// - barycentric 双向映射
	//
	// 执行顺序：
	// 1. 先细分 surface_infile，得到下一轮 surface_outfile，同时收集当前边界边；
	// 2. 再按内部边 shard 收集 edge request；
	// 3. 逐 shard 去重并一次性创建内部边中点；
	// 4. 最后第二遍读取 tet_infile，输出下一轮 tet_outfile。
	//
	// 这样就把“表面 -> 体细化”的桥梁从内存 newfaces 改成了
	// “当前 submesh + 外存 stream + 双向 barycentric 映射”。
	const std::size_t shard_count = edge_shards == 0 ? 16 : edge_shards;
	std::size_t total_unique_internal_edges = 0;

	// 阶段1：surface refine 保持原样
	PrintInnerMemLog("enter", round, output_path, submesh, baryc2locvrtxmap, locvrtx2barycmap, edgemap);
	edgemap.clear();

	std::set<IntPair, IntPairCompare> boundary_edges;
	StreamRefineFacesToFile(submesh,
		surface_infile,
		surface_outfile,
		batch_faces,
		baryc2locvrtxmap,
		locvrtx2barycmap,
		edgemap,
		&boundary_edges,
		point_table_path,
		next_point_id,
		false);
	const std::size_t boundary_edges_size = boundary_edges.size();
	PrintInnerMemLog("after_surface", round, output_path, submesh, baryc2locvrtxmap, locvrtx2barycmap, edgemap, &boundary_edges_size, nullptr);

	// 阶段2：按 shard 收集内部边请求
	const std::vector<std::string> edge_req_files = BuildInteriorEdgeShardFiles(
		tet_infile,
		tet_outfile,
		boundary_edges,
		batch_tets,
		round,
		shard_count);
	PrintInnerMemLog("after_edge_req_shards", round, output_path, submesh, baryc2locvrtxmap, locvrtx2barycmap, edgemap, nullptr, nullptr, &total_unique_internal_edges);

	// 阶段3：逐 shard 去重并创建内部边中点
	const std::vector<std::string> edge_mid_files = BuildEdgeShardFilePaths(tet_outfile, "edge_mid", round, shard_count);
	const std::vector<std::vector<EdgeMidRecord>> edge_mid_shards = BuildEdgeMidShardTables(
		edge_req_files,
		edge_mid_files,
		point_table_path,
		batch_tets,
		next_point_id,
		total_unique_internal_edges);
	PrintInnerMemLog("after_edge_mid_shards", round, output_path, submesh, baryc2locvrtxmap, locvrtx2barycmap, edgemap, nullptr, nullptr, &total_unique_internal_edges);

	// 阶段4：第二遍读取 tet_infile，输出 refined tets
	std::ifstream input(tet_infile, std::ios::binary);
	if (!input)
	{
		FailStreamOperation("failed to open tet stream input file: " + tet_infile);
	}
	EnsureParentDirectory(tet_outfile);
	std::ofstream output(tet_outfile, std::ios::binary | std::ios::trunc);
	if (!output)
	{
		FailStreamOperation("failed to open tet stream output file: " + tet_outfile);
	}

	std::vector<TetRecord> input_buffer;
	std::vector<TetRecord> output_buffer;
	output_buffer.reserve(EffectiveBatch(batch_tets) * 8);
	std::size_t batch_count = 0;

	while (ReadTetBatch(input, input_buffer, batch_tets) > 0)
	{
		++batch_count;
		for (const TetRecord &record : input_buffer)
		{
			int Ev[10] = {0};
			int domainidx = 0;
			FromTetRecord(record, Ev, domainidx);

			for (int edge = 0; edge < 6; ++edge)
			{
				const int va = Ev[kTetEdges[edge][0]];
				const int vb = Ev[kTetEdges[edge][1]];
				const IntPair ordered_edge = MakeOrderedEdge(va, vb);

				if (boundary_edges.find(ordered_edge) != boundary_edges.end())
				{
					Barycentric bary0{};
					Barycentric bary1{};
					if (!TryGetVertexBarycentric(va, locvrtx2barycmap, bary0) || !TryGetVertexBarycentric(vb, locvrtx2barycmap, bary1))
					{
						FailStreamOperation("missing boundary barycentric mapping while refining volume stream");
					}
					Ev[edge + 4] = GetOrCreateEdgeMidpoint((nglib::Ng_Mesh *)submesh,
						va,
						vb,
						&bary0,
						&bary1,
						baryc2locvrtxmap,
						locvrtx2barycmap,
						edgemap,
						nullptr,
						point_table_path,
						next_point_id,
						false);
				}
				else
				{
					const EdgeKey key = PackEdgeKey(ordered_edge.x, ordered_edge.y);
					const std::size_t shard = EdgeShardIndex(key, shard_count);
					Ev[edge + 4] = LookupMidpointInShardVector(edge_mid_shards[shard], key);
				}
			}

			for (int j = 0; j < 8; ++j)
			{
				int pointindex[4];
				for (int k = 0; k < 4; ++k)
				{
					pointindex[k] = Ev[kTetRefTab[j][k]];
				}
				output_buffer.push_back(ToTetRecord(pointindex, domainidx));
			}

			if (output_buffer.size() >= EffectiveBatch(batch_tets) * 8)
			{
				FlushTetBuffer(output, output_buffer);
			}
		}

		if (IsPowerOfTwo(batch_count))
		{
			const std::size_t output_buffer_size = output_buffer.size();
			PrintInnerMemLog("tet_loop", round, output_path, submesh, baryc2locvrtxmap, locvrtx2barycmap, edgemap, nullptr, &output_buffer_size);
		}
	}

	FlushTetBuffer(output, output_buffer);
	PrintInnerMemLog("exit", round, output_path, submesh, baryc2locvrtxmap, locvrtx2barycmap, edgemap);

	for (const std::string &filepath : edge_req_files)
	{
		std::filesystem::remove(filepath);
	}
	for (const std::string &filepath : edge_mid_files)
	{
		std::filesystem::remove(filepath);
	}
}

void *ReplayTetsFromFileToMesh(void *submesh, const std::string &filepath, std::size_t batch_tets)
{
	// 最后一轮结束后，再把最终 tet stream 回灌到 mesh。
	//
	// 为什么不在每轮中途就回灌？
	// - 中间轮 tet 数量会快速膨胀；
	// - 如果每轮都长期挂在 submesh 里，峰值内存会重新上去；
	// - 因此 stream 路径把体单元的常驻时机推迟到最后一步。
	//
	// 这里先构造一个“只保留点、face descriptor 和 surface elements”的新 mesh，
	// 再把最终 tet 文件顺序回放进去。
	nglib::Ng_Mesh *mesh = ClearVolumeElementsOrEquivalent((nglib::Ng_Mesh *)submesh);

	std::ifstream input(filepath, std::ios::binary);
	if (!input)
	{
		FailStreamOperation("failed to open tet replay file: " + filepath);
	}

	std::vector<TetRecord> buffer;
	while (ReadTetBatch(input, buffer, batch_tets) > 0)
	{
		for (const TetRecord &record : buffer)
		{
			int vids[4];
			int domidx = 0;
			FromTetRecord(record, vids, domidx);
			nglib::Ng_AddVolumeElement(mesh, nglib::NG_TET, vids, domidx);
		}
	}

	return mesh;
}

void Refineforvol(void *submesh, int belongNumberPartition, std::list<xdFace> &newfaces, std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap, std::map<int, Barycentric> &locvrtx2barycmap, std::map<IntPair, int, IntPairCompare> &edgemap)
{
	// 原版非流式 volume refine 路径保留，用于功能对照。
	// 逻辑仍然是：
	// 1. 先细分 newfaces 并重建 surface；
	// 2. 再对当前 mesh 中每个四面体做 8-tet 细分。
	//
	// 与旧代码相比，这里补齐了 locvrtx2barycmap 的维护能力，
	// 但总体算法和细分策略不变。
	(void)belongNumberPartition;
	const int oldnse = nglib::Ng_GetNSE((nglib::Ng_Mesh *)submesh);
	std::list<xdFace>::iterator li = newfaces.begin();
	std::vector<xdFace> children;
	children.reserve(4);

	for (int i = 0; i < oldnse; ++i)
	{
		const xdFace face = *li;
		RefineFaceAndCollectChildren((nglib::Ng_Mesh *)submesh, face, children, baryc2locvrtxmap, locvrtx2barycmap, edgemap, nullptr);
		for (const xdFace &child : children)
		{
			newfaces.push_back(child);
		}
		li = newfaces.erase(li);
	}

	nglib::ClearSurfaceElements((nglib::Ng_Mesh *)submesh);
	for (li = newfaces.begin(); li != newfaces.end(); ++li)
	{
		nglib::Ng_AddSurfaceElementwithIndex((nglib::Ng_Mesh *)submesh, nglib::NG_TRIG, (*li).lsvrtx, (*li).geoboundary);
	}

	const int oldne = nglib::Ng_GetNE((nglib::Ng_Mesh *)submesh);
	for (int i = 0; i < oldne; ++i)
	{
		int Ev[10] = {0};
		int domainidx = 0;
		nglib::Ng_GetVolumeElement((nglib::Ng_Mesh *)submesh, i + 1, Ev, domainidx);
		for (int edge = 0; edge < 6; ++edge)
		{
			Ev[edge + 4] = GetOrCreateEdgeMidpoint((nglib::Ng_Mesh *)submesh, Ev[kTetEdges[edge][0]], Ev[kTetEdges[edge][1]], nullptr, nullptr, baryc2locvrtxmap, locvrtx2barycmap, edgemap, nullptr);
		}

		int pointindex[4];
		for (int j = 0; j < 8; ++j)
		{
			for (int k = 0; k < 4; ++k)
			{
				pointindex[k] = Ev[kTetRefTab[j][k]];
			}
			if (j == 0)
			{
				nglib::My_Ng_SetVolumeElement((nglib::Ng_Mesh *)submesh, i + 1, pointindex, domainidx);
			}
			else
			{
				nglib::Ng_AddVolumeElement((nglib::Ng_Mesh *)submesh, nglib::NG_TET, pointindex, domainidx);
			}
		}
	}
}

//将顶点和面的关系添加到一个映射中
void insbaryadjlist(
	Barycvrtx bvrtx,
	int pid,
	std::map<Barycvrtx, std::list<int>, CompBarycvrtx> &barycvrtx2adjprocsmap)
{
	std::map<Barycvrtx, std::list<int>, CompBarycvrtx>::iterator ivrtx;
	std::list<int> pidlist;
	std::list<int>::iterator li;
	// printf("%d %d\n" ,vrtx, pid) ;
	ivrtx = barycvrtx2adjprocsmap.find(bvrtx);
	if (ivrtx == barycvrtx2adjprocsmap.end())
	{
		barycvrtx2adjprocsmap[bvrtx] = pidlist;
		barycvrtx2adjprocsmap[bvrtx].push_back(pid);
	}
	else
	{
		for (li = (ivrtx->second).begin(); li != (ivrtx->second).end(); ++li)
		{
			if ((*li) == pid)
				break;
		}
		if (li == (ivrtx->second).end())
		{
			barycvrtx2adjprocsmap[bvrtx].push_back(pid);
		}
	}
}

//在 MPI 通信中发送和接收数据
int com_sr_datatype(
	MPI_Comm comm,
	int num_s,
	int num_r,
	int *dest,
	int *src,
	int *s_length,
	int *r_length,
	Barycentric **s_data,
	Barycentric **r_data,
	MPI_Datatype datatype,
	int mypid)
{
	int i;
	MPI_Status *stat;
	MPI_Request *req;
	int rc;
	if ((num_s + num_r) > 0)
	{
		MYCALLOC(req, MPI_Request *, (num_s + num_r), sizeof(MPI_Request));
		MYCALLOC(stat, MPI_Status *, (num_s + num_r), sizeof(MPI_Status));
	}
	else
	{
		return (MPI_SUCCESS);
	}
	for (i = 0; i < num_s; i++)
	{
		rc = MPI_Isend(s_data[i], s_length[i], datatype, dest[i], mypid, comm,
					   &(req[i]));
		if (rc != MPI_SUCCESS)
		{
			printf("Error in mpi\n");
			exit(1);
		}
	}
	for (i = 0; i < num_r; i++)
	{
		rc = MPI_Irecv(r_data[i], r_length[i], datatype, src[i], src[i], comm,
					   &(req[num_s + i]));
		if (rc != MPI_SUCCESS)
		{
			printf("Error in mpi\n");
			exit(1);
		}
	}
	if (num_s + num_r)
	{
		rc = MPI_Waitall(num_s + num_r, req, stat);
		if (rc != MPI_SUCCESS)
		{
			printf("Error in mpi\n");
			exit(1);
		}
	}
	free(req);
	free(stat);
	return (rc);
}

//在 MPI 通信中发送和接收整数数据
int com_sr_int(
	MPI_Comm comm,
	int num_s,
	int num_r,
	int *dest,
	int *src,
	int **s_data,
	int **r_data,
	int mypid)
{
	int i;
	MPI_Status *stat;
	MPI_Request *req;
	int rc;
	if ((num_s + num_r) > 0)
	{
		MYCALLOC(req, MPI_Request *, (num_s + num_r), sizeof(MPI_Request));
		MYCALLOC(stat, MPI_Status *, (num_s + num_r), sizeof(MPI_Status));
	}
	else
	{
		return (MPI_SUCCESS);
	}
	for (i = 0; i < num_s; i++)
	{
		rc = MPI_Isend(s_data[i], 1, MPI_INT, dest[i], mypid, comm,
					   &(req[i]));
		if (rc != MPI_SUCCESS)
		{
			printf("Error in mpi\n");
			exit(1);
		}
	}
	for (i = 0; i < num_r; i++)
	{
		rc = MPI_Irecv(r_data[i], 1, MPI_INT, src[i], src[i], comm,
					   &(req[num_s + i]));
		if (rc != MPI_SUCCESS)
		{
			printf("Error in mpi\n");
			exit(1);
		}
	}
	if (num_s + num_r)
	{
		rc = MPI_Waitall(num_s + num_r, req, stat);
		if (rc != MPI_SUCCESS)
		{
			printf("Error in mpi\n");
			exit(1);
		}
	}
	free(req);
	free(stat);
	return (rc);
}

//在 MPI 通信中发送和接收 xdVElement 结构体数据
int com_sr_volumelement(
	MPI_Comm comm,
	int num_s,
	int num_r,
	int *dest,
	int *src,
	int *s_length,
	int *r_length,
	xdVElement **s_data,
	xdVElement **r_data,
	MPI_Datatype datatype,
	int mypid)
{
	int i;
	MPI_Status *stat;
	MPI_Request *req;
	int rc;
	if ((num_s + num_r) > 0)
	{
		MYCALLOC(req, MPI_Request *, (num_s + num_r), sizeof(MPI_Request));
		MYCALLOC(stat, MPI_Status *, (num_s + num_r), sizeof(MPI_Status));
	}
	else
	{
		return (MPI_SUCCESS);
	}
	for (i = 0; i < num_s; i++)
	{
		rc = MPI_Isend(s_data[i], s_length[i], datatype, dest[i], mypid, comm,
					   &(req[i]));
		if (rc != MPI_SUCCESS)
		{
			printf("Error in mpi\n");
			exit(1);
		}
	}
	for (i = 0; i < num_r; i++)
	{
		rc = MPI_Irecv(r_data[i], r_length[i], datatype, src[i], src[i], comm,
					   &(req[num_r + i]));
		if (rc != MPI_SUCCESS)
		{
			printf("Error in mpi\n");
			exit(1);
		}
	}
	if (num_s + num_r)
	{
		rc = MPI_Waitall(num_s + num_r, req, stat);
		if (rc != MPI_SUCCESS)
		{
			printf("Error in mpi\n");
			exit(1);
		}
	}
	free(req);
	free(stat);
	return (rc);
}


//函数会计算出每个顶点与其邻接进程之间的关系，并将结果存储在 barycvrtx2adjprocsmap 中。
void computeadj(
	int mypid, std::map<int, xdMeshFaceInfo> &facemap, std::map<int, int> &g2lvrtxmap,
	std::map<Barycvrtx, std::list<int>, CompBarycvrtx> &barycvrtx2adjprocsmap)
{
	std::map<int, xdMeshFaceInfo>::iterator itf;
	std::map<int, int>::iterator itv;
	int fid;
	xdMeshFaceInfo finfo;
	int pid, i;
	std::map<Barycvrtx, std::list<int>, CompBarycvrtx>::iterator ib; // barycvrtx ȫ�ֶ���id������(��1��ʼ)
	std::list<int>::iterator li;
	int count;
	Barycvrtx bvrtx;
	Barycvrtx countbvrtx;

	for (itf = facemap.begin(); itf != facemap.end(); ++itf)
	{
		fid = itf->first;
		finfo = itf->second;
		if ((finfo.procids[0] != mypid) && (finfo.procids[1] == -1))
			continue;
		if ((finfo.procids[1] != mypid) && (finfo.procids[0] == -1))
			continue;
		if (finfo.procids[0] == finfo.procids[1])
			continue;
		for (int j = 0; j < 2; j++)
		{
			pid = finfo.procids[j];
			if ((pid == mypid) || (pid == -1))
				continue; // �ҽ����� pid != mypid II pid != 1��������ִ��,����ִ����һ��ѭ��
			count = 0;
			for (int k = 0; k < 3; k++)
			{
				countbvrtx.gvrtx[k] = 0;
				itv = g2lvrtxmap.find(finfo.svrtx[j][k]);
				if (itv != g2lvrtxmap.end())
				{
					bvrtx.gvrtx[0] = 0;
					bvrtx.gvrtx[1] = 0;
					bvrtx.gvrtx[2] = finfo.svrtx[j][k];
					countbvrtx.gvrtx[k] = finfo.svrtx[j][k]; // ���ϵ������
					count++;
					insbaryadjlist(bvrtx, pid, barycvrtx2adjprocsmap); // ��������ĵ���������ǩ
				}
			}
			sort3int(countbvrtx.gvrtx);
			if (count == 2)
			{															// insert edge adjacency
				insbaryadjlist(countbvrtx, pid, barycvrtx2adjprocsmap); // ��������.�ϵı߼��������ǩ
			}
			if (count == 3)
			{ // insert edge and face adjacencies ���� �ߺ����ڽ�
				bvrtx.gvrtx[0] = 0;
				bvrtx.gvrtx[1] = countbvrtx.gvrtx[0];
				bvrtx.gvrtx[2] = countbvrtx.gvrtx[1];
				insbaryadjlist(bvrtx, pid, barycvrtx2adjprocsmap);
				bvrtx.gvrtx[0] = 0;
				bvrtx.gvrtx[1] = countbvrtx.gvrtx[0];
				bvrtx.gvrtx[2] = countbvrtx.gvrtx[2];
				insbaryadjlist(bvrtx, pid, barycvrtx2adjprocsmap);
				bvrtx.gvrtx[0] = 0;
				bvrtx.gvrtx[1] = countbvrtx.gvrtx[1];
				bvrtx.gvrtx[2] = countbvrtx.gvrtx[2];
				insbaryadjlist(bvrtx, pid, barycvrtx2adjprocsmap);
				insbaryadjlist(countbvrtx, pid, barycvrtx2adjprocsmap); // face adjacency�� �ڽ�
			}
		}
	}
}

void DumpAdjBarycsSummary(
	const std::string &filepath,
	int mypid,
	const std::map<Barycvrtx, std::list<int>, CompBarycvrtx> &barycvrtx2adjprocsmap)
{
	std::ofstream output(filepath, std::ios::app);
	if (!output)
	{
		FailStreamOperation("failed to open adjacency summary file: " + filepath);
	}

	output << "rank=" << mypid << "\n";
	output << "adj_entry_count=" << barycvrtx2adjprocsmap.size() << "\n";

	int item_index = 0;
	for (const auto &entry : barycvrtx2adjprocsmap)
	{
		if (item_index >= 10)
		{
			break;
		}
		output << "item" << item_index << "="
		       << entry.first.gvrtx[0] << ","
		       << entry.first.gvrtx[1] << ","
		       << entry.first.gvrtx[2] << " -> ";
		bool first_pid = true;
		for (const int pid : entry.second)
		{
			if (!first_pid)
			{
				output << ",";
			}
			output << pid;
			first_pid = false;
		}
		output << "\n";
		++item_index;
	}
	output << "\n";
}

int *com_barycoords(
	void *submesh,
	MPI_Comm comm,
	std::map<Barycvrtx, std::list<int>, CompBarycvrtx> &barycvrtx2adjprocsmap,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, std::list<int>> &adjbarycs,
	int numprocs, int *newgVEid,
	int mypid)
{
	int count = 3;
	int blocklens[3];
	MPI_Aint addrs[3];
	MPI_Datatype mpitypes[3];
	MPI_Datatype mpibaryctype;
	int num_s, num_r; // number of sends and receives ���ͺͽ��յ�����
	int *dest, *src;

	int *s_length, *r_length;
	// sent/received data size

	Barycentric **s_data, **r_data; // sent / received data

	Barycentric brcy;
	// struct used to get struct member

	std::map<Barycvrtx, std::list<int>>::iterator ibc;
	std::list<int>::iterator li;
	std::map<Barycentric, int, CompBarycentric>::iterator ib;
	Baryvrtx bvrtx;
	std::map<int, BarycVector *>::iterator ipdt;
	std::map<int, int>::iterator srcit; // new
	int i, j, indxowner, ownerpid, locid;
	std::vector<int> holders;
	int numverts, numNEs;
	int *globoffsets, *globoffsetsml;
	int *globoffsetsVE, *globoffsetsmlVE;
	std::map<int, BarycVector *> pidmap;
	std::map<int, int> srcmap; // new
	int newglobalnocounter = 0;
	int newglobalVEocounter = 0;
	int *newgid;
	std::list<int> pids;
	// initialize new global ids array ��ʼ���µ�ȫ��id����
	numverts = nglib::Ng_GetNP((nglib::Ng_Mesh *)submesh);
	numNEs = nglib::Ng_GetNE((nglib::Ng_Mesh *)submesh);
	MYCALLOC(newgid, int *, (numverts + 1), sizeof(int)); // ids start with 1,initialized to 0 ids��1��ʼ, ��ʼ��Ϊ0
	num_s = 0;
	num_r = 0;
	// construct MPI Datatype MPI������������
	blocklens[0] = 3;
	blocklens[1] = 3;
	blocklens[2] = 1;
	mpitypes[0] = MPI_INT;
	mpitypes[1] = MPI_SHORT;
	mpitypes[2] = MPI_INT;
	//MPI_Address(&brcy.gvrtx, addrs);
	//MPI_Address(&brcy.coord, addrs + 1);
	//MPI_Address(&brcy.newgid, addrs + 2);
    MPI_Get_address(&brcy.gvrtx, addrs);
	MPI_Get_address(&brcy.coord, addrs + 1);
	MPI_Get_address(&brcy.newgid, addrs + 2);

	addrs[1] = addrs[1] - addrs[0];
	addrs[2] = addrs[2] - addrs[0];
	addrs[0] = (MPI_Aint)0;
	//MPI_Type_struct(count, blocklens, addrs, mpitypes, &mpibaryctype);
    MPI_Type_create_struct(count, blocklens, addrs, mpitypes, &mpibaryctype);
	MPI_Type_commit(&mpibaryctype);
	newglobalnocounter = 0; // initialize global number counter��ʼ��ȫ �����ּ�����
	// Loop goes over the geometric and partition boundary vertices in the new meshѭ���� ���������еļ��κͻ��ֱ߽綥��
	int length = 0;
	for (ib = baryc2locvrtxmap.begin(); ib != baryc2locvrtxmap.end(); ++ib)
	{
		brcy = ib->first;
		locid = ib->second;

		// ���߽綥�������ȫ�ֶ�������.
		bvrtx.gvrtx[0] = brcy.gvrtx[0];
		bvrtx.gvrtx[1] = brcy.gvrtx[1];
		bvrtx.gvrtx[2] = brcy.gvrtx[2];
		ibc = barycvrtx2adjprocsmap.find(bvrtx);
		if (ibc == barycvrtx2adjprocsmap.end())
		{
			continue;
		}
		length++;
		pids = ibc->second;
		adjbarycs[locid] = pids;
		// compute owners of shared vertices(held by multiple processors - called holders) ������ ��ļ���������(�ɶ������������ - -��Ϊ������)
		holders.clear();
		holders.push_back(mypid); // i am also a holder
		for (li = (ibc->second).begin(); li != (ibc->second).end(); ++li)
		{
			holders.push_back((*li));
		}

		std::sort(holders.begin(), holders.end());
		indxowner = (brcy.gvrtx[0] + brcy.gvrtx[1] + brcy.gvrtx[2] +
					 brcy.coord[0] + brcy.coord[1] + brcy.coord[2]) %
					(ibc->second).size();
		ownerpid = holders[indxowner]; // if I am the owner, append it to the message to be sent �������������,���丽�ӵ�Ҫ���͵���Ϣ��
		if (ownerpid == mypid)
		{
			newglobalnocounter++;
			newgid[locid] = newglobalnocounter;
			brcy.newgid = newglobalnocounter;
			for (li = (ibc->second).begin(); li != (ibc->second).end(); ++li)
			{
				ipdt = pidmap.find((*li));
				if (ipdt == pidmap.end())
				{
					pidmap[(*li)] = new BarycVector();
				}
				pidmap[(*li)]->push_back(brcy);
			}
		}
		else
		{ // i am not the owner of this partition boundary vertex �Ҳ�������ֿ�߽綥���������
			newgid[locid] = -1;
			srcit = srcmap.find(ownerpid);
			if (srcit == srcmap.end())
			{
				srcmap[ownerpid] = 1;
			}
			else
			{
				srcit->second = srcit->second + 1;
			}
		}
	}
	for (locid = 1; locid <= numverts; locid++)
	{
		if (newgid[locid] == 0)
		{
			newglobalnocounter++;
			newgid[locid] = newglobalnocounter;
		}
	}
	// compute pre - scan of all newglobalnocounter in array globoffsets
	MYCALLOC(globoffsetsml, int *, (numprocs + 1), sizeof(int));
	globoffsets = globoffsetsml + 1;
	globoffsets[-1] = 0;
	MPI_Allgather(&newglobalnocounter, 1, MPI_INT, globoffsets, 1, MPI_INT, comm);
	for (i = 0; i < numprocs; i++)
		globoffsets[i] += globoffsets[i - 1];
	// add offsets to global numbers
	// new
	for (locid = 1; locid <= numverts; locid++)
	{
		if (newgid[locid] != -1)
		{
			newgid[locid] += globoffsets[mypid - 1];
		}
	}
	// now do global numbering of the interior vertices ���ڶ��ڲ��嵥Ԫ����ȫ�ֱ��
	int locVEid;
	// compute pre - scan of all newglobalnocounter in array globoffsets
	MYCALLOC(globoffsetsmlVE, int *, (numprocs + 1), sizeof(int));
	globoffsetsVE = globoffsetsmlVE + 1;
	globoffsetsVE[-1] = 0;
	MPI_Allgather(&numNEs, 1, MPI_INT, globoffsetsVE, 1, MPI_INT, comm);
	for (i = 0; i < numprocs; i++)
		globoffsetsVE[i] += globoffsetsVE[i - 1];
	// add offsets to global numbers
	// new
	for (locVEid = 1; locVEid <= numNEs; locVEid++)
	{
		newgVEid[locVEid] += locVEid + globoffsetsVE[mypid - 1];
	}
	// new
	num_s = pidmap.size();
	if (num_s > 0)
	{
		MYCALLOC(s_length, int *, num_s, sizeof(int));
		MYCALLOC(dest, int *, num_s, sizeof(int));
		MYCALLOC(s_data, Barycentric **, num_s, sizeof(Barycentric *));
	}
	else
	{
		dest = nullptr;
		s_length = nullptr;
		s_data = nullptr;
	}
	i = 0;
	for (ipdt = pidmap.begin(); ipdt != pidmap.end(); ++ipdt)
	{
		s_length[i] = ipdt->second->size();
		dest[i] = ipdt->first;
		s_data[i] = &((*ipdt->second)[0]);
		i++;
	}
	num_r = srcmap.size();
	if (num_r > 0)
	{
		MYCALLOC(r_length, int *, num_r, sizeof(int));
		MYCALLOC(src, int *, num_r, sizeof(int));
		MYCALLOC(r_data, Barycentric **, num_r, sizeof(Barycentric *));
	}
	else
	{
		src = nullptr;
		r_length = nullptr;
		r_data = nullptr;
	}
	// compute lengths of messages (no. of items) that will be sent
	// new
	for (srcit = srcmap.begin(), i = 0; srcit != srcmap.end(); ++srcit, ++i)
	{
		src[i] = srcit->first;
		r_length[i] = srcit->second;
	}
	// new
	for (i = 0; i < num_r; i++)
	{
		MYCALLOC(r_data[i], Barycentric *, r_length[i], sizeof(Barycentric));
	}
	com_sr_datatype(comm, num_s, num_r, dest, src, s_length, r_length,
					s_data, r_data, mpibaryctype, mypid);
	for (i = 0; i < num_r; i++)
	{
		for (j = 0; j < r_length[i]; j++)
		{
			locid = baryc2locvrtxmap[r_data[i][j]];
			if (newgid[locid] != -1)
				printf("%d> Error: remote global id %d\n", mypid, locid);
			// newgid[locid] = -(r_ data[i][j]. newgid + globoffsets[src[i] - 1]);
			newgid[locid] = (r_data[i][j].newgid + globoffsets[src[i] - 1]);
		}
	}
	for (i = 0; i < num_r; i++)
	{
		free(r_data[i]);
	}
	if (num_s > 0)
	{
		free(s_length);
		free(s_data);
		free(dest);
	}
	if (num_r > 0)
	{
		free(r_length);
		free(r_data);
		free(src);
	}
	free(globoffsetsml);
	free(globoffsetsmlVE);
	MPI_Type_free(&mpibaryctype);
	return newgid;
}

// 该函数用于在并行环境下通信和构建体单元（Volume Elements）
// 将相邻进程需要的体单元发送给对应进程，并在本地网格中补充缺失的点和单元
int *com_baryVolumeElements(
	void *submesh,   // 子网格
	MPI_Comm comm,   // MPI通信域
	std::map<Barycvrtx, std::list<int>, CompBarycvrtx> &barycvrtx2adjprocsmap,
	std::map<Barycentric, int, CompBarycentric> &baryc2locvrtxmap,
	std::map<int, std::list<int>> &adjbarycs,  // barycentric邻接关系
	int *oldgid,     // 原始点的全局ID
	int *VEgid,      // 体单元全局ID
	std::list<VEindex> &VEindexs, // 体单元索引列表
	int numprocs,    // 总进程数
	int mypid)       // 当前进程ID
{
	// 将输入的void指针转换为Netgen网格对象
	nglib::Ng_Mesh *mesh = (nglib::Ng_Mesh *)submesh;

	int locid;
	std::list<int> pids;
	std::list<int>::iterator li;

	// 邻接barycentric迭代器
	std::map<int, std::list<int>>::iterator ib;

	// 用于记录需要发送到哪个进程的体单元
	std::map<int, VEVector *> pidmap;
	std::map<int, VEVector *>::iterator ipm;

	// 获取当前网格中的体单元数量
	int num_keys = nglib::Ng_GetNE(mesh);

	std::set<int> pid_tmp;   // 存储该体单元相邻的进程ID
	std::set<int> vols_tmp;  // 临时存储体单元集合（防止重复）

	std::set<int>::iterator pt;

	int i;
	int domainidx;

	// 存储点坐标
	double xyz[3];

	// 体单元结构体
	xdVElement ve;

	// =============================
	// 构造MPI数据类型（用于发送体单元）
	// =============================

	int count = 4;  // 结构体成员数量

	int blocklens[4];  // 每个成员的长度
	MPI_Aint addrs[4]; // 地址偏移
	MPI_Datatype mpitypes[4]; // 对应MPI类型
	MPI_Datatype mpivetype;

	// Pindex[4]
	blocklens[0] = 4;

	// Vertexs[4] 每个点3坐标，共12个double
	blocklens[1] = 12;

	// gid
	blocklens[2] = 1;

	// domain index
	blocklens[3] = 1;

	mpitypes[0] = MPI_INT;
	mpitypes[1] = MPI_DOUBLE;
	mpitypes[2] = MPI_INT;
	mpitypes[3] = MPI_INT;

	// 获取结构体成员地址
	MPI_Get_address(&ve.Pindex, addrs);
	MPI_Get_address(&ve.Vertexs, addrs + 1);
	MPI_Get_address(&ve.gid, addrs + 2);
	MPI_Get_address(&ve.domidx, addrs + 3);

	// 转换为相对地址
	addrs[3] = addrs[3] - addrs[0];
	addrs[2] = addrs[2] - addrs[0];
	addrs[1] = addrs[1] - addrs[0];
	addrs[0] = (MPI_Aint)0;

	// 创建MPI结构体类型
	MPI_Type_create_struct(count, blocklens, addrs, mpitypes, &mpivetype);
	MPI_Type_commit(&mpivetype);

	int entity[4];  // 四面体的4个顶点
	int *vols;
	int volsize;

	// =============================
	// 遍历所有体单元
	// =============================

	for (i = 0; i < num_keys; i++)
	{
		// 获取第i个体单元的顶点
		nglib::Ng_GetVolumeElement(mesh, i + 1, entity, domainidx);

		pid_tmp.clear();

		// 记录与多少邻接点相连
		std::map<int, int> num_adjPoints;
		std::map<int, int>::iterator igap;

		num_adjPoints.clear();

		// 遍历四面体4个顶点
		for (int k = 0; k < 4; k++)
		{
			ib = adjbarycs.find(entity[k]);
			if (ib != adjbarycs.end())
			{
				for (li = (ib->second).begin(); li != (ib->second).end(); ++li)
				{
					num_adjPoints[*li]++;
				}
			}
		}

		// 如果某个进程共享>=3个顶点，则认为需要共享该体单元
		for (igap = num_adjPoints.begin(); igap != num_adjPoints.end(); ++igap)
		{
			if (igap->second >= 3)
			{
				pid_tmp.insert(igap->first);
			}
		}

		// 如果存在需要共享的进程
		if (pid_tmp.size())
		{
			// 记录该体单元信息
			for (int k = 0; k < 4; k++)
			{
				nglib::Ng_GetPoint(mesh, entity[k], xyz);

				ve.Vertexs[k].xyz[0] = xyz[0];
				ve.Vertexs[k].xyz[1] = xyz[1];
				ve.Vertexs[k].xyz[2] = xyz[2];

				// 全局点ID
				ve.Pindex[k] = oldgid[entity[k]];
			}

			ve.domidx = domainidx;
			ve.gid = VEgid[i + 1];

			// 添加到对应进程的发送列表
			for (pt = pid_tmp.begin(); pt != pid_tmp.end(); ++pt)
			{
				ipm = pidmap.find((*pt));

				if (ipm == pidmap.end())
				{
					pidmap[(*pt)] = new VEVector();
				}

				pidmap[(*pt)]->push_back(ve);
			}
		}
	}

	// =============================
	// 准备MPI通信
	// =============================

	int num_s, num_r; // 发送数量和接收数量

	num_s = num_r = pidmap.size();

	int *dest, *src;
	int **s_data, **r_data, *size;

	if (num_s > 0)
	{
		MYCALLOC(dest, int *, num_s, sizeof(int));
		MYCALLOC(s_data, int **, num_s, sizeof(int *));
		MYCALLOC(src, int *, num_s, sizeof(int));
		MYCALLOC(r_data, int **, num_s, sizeof(int *));
		MYCALLOC(size, int *, num_s, sizeof(int));
	}
	else
	{
		dest = nullptr;
		s_data = nullptr;
		src = nullptr;
		r_data = nullptr;
		size = nullptr;
	}

	// 初始化接收缓存
	for (i = 0; i < num_r; i++)
	{
		MYCALLOC(r_data[i], int *, 1, sizeof(int));
	}

	i = 0;

	// 填充通信信息
	for (ipm = pidmap.begin(); ipm != pidmap.end(); ++ipm)
	{
		dest[i] = ipm->first;
		src[i] = ipm->first;

		size[i] = ipm->second->size();

		s_data[i] = &size[i];

		i++;
	}

	// 通信发送每个进程需要发送的数据大小
	com_sr_int(comm, num_s, num_r, dest, src, s_data, r_data, mypid);

	// =============================
	// 发送体单元数据
	// =============================

	xdVElement **r_data_ves, **s_data_ves;
	int *r_length, *s_length;

	if (num_s > 0)
	{
		MYCALLOC(s_length, int *, num_s, sizeof(int));
		MYCALLOC(s_data_ves, xdVElement **, num_s, sizeof(xdVElement *));
		MYCALLOC(r_length, int *, num_r, sizeof(int));
		MYCALLOC(r_data_ves, xdVElement **, num_r, sizeof(xdVElement *));
	}
	else
	{
		r_length = nullptr;
		r_data_ves = nullptr;
		s_length = nullptr;
		s_data_ves = nullptr;
	}

	i = 0;

	for (ipm = pidmap.begin(); ipm != pidmap.end(); ++ipm)
	{
		s_length[i] = *s_data[i];
		r_length[i] = *r_data[i];

		s_data_ves[i] = &((*ipm->second)[0]);

		i++;
	}

	// 为接收数据分配空间
	for (i = 0; i < num_r; i++)
	{
		MYCALLOC(r_data_ves[i], xdVElement *, r_length[i], sizeof(xdVElement));
	}

	// 体单元通信
	com_sr_volumelement(comm, num_s, num_r, dest, src, s_length, r_length,
						s_data_ves, r_data_ves, mpivetype, mypid);

	// =============================
	// 将接收到的体单元加入本地网格
	// =============================

	int oldpointnum;
	oldpointnum = nglib::Ng_GetNP(mesh);

	std::map<int, int> gid2lid; // global id -> local id
	std::map<int, int>::iterator ig2l;

	std::map<int, int> gids_add;
	std::map<int, int>::iterator iga;

	std::map<int, int> gidVEs_add;
	std::map<int, int>::iterator igaVE;

	int index = -1;
	int pi[4];

	// 建立gid到local id的映射
	for (i = 0; i < nglib::Ng_GetNP(mesh); i++)
	{
		gid2lid[oldgid[i]] = i;
	}

	int numVEcount, numVEold;

	numVEcount = nglib::Ng_GetNE(mesh);
	numVEold = nglib::Ng_GetNE(mesh);

	// 处理接收的体单元
	for (i = 0; i < num_r; i++)
	{
		for (int j = 0; j < r_length[i]; j++)
		{
			ve = r_data_ves[i][j];

			for (int k = 0; k < 4; k++)
			{
				ig2l = gid2lid.find(ve.Pindex[k]);

				if (ig2l == gid2lid.end())
				{
					// 如果点不存在，则添加
					nglib::Ng_AddPoint(mesh, ve.Vertexs[k].xyz, index);

					gids_add[index] = ve.Pindex[k];
					gid2lid[ve.Pindex[k]] = index;

					pi[k] = index;
				}
				else
				{
					pi[k] = ig2l->second;
				}
			}

			// 添加体单元
			nglib::Ng_AddVolumeElement(mesh, nglib::NG_TET, pi, ve.domidx);

			numVEcount++;

			gidVEs_add[numVEcount] = ve.gid;
		}
	}

	// =============================
	// 构建新的全局ID数组
	// =============================

	int *newgid;

	MYCALLOC(newgid, int *, nglib::Ng_GetNP(mesh) + 1, sizeof(int));

	for (i = 1; i < oldpointnum + 1; i++)
		newgid[i] = oldgid[i];

	for (iga = gids_add.begin(); iga != gids_add.end(); ++iga)
		newgid[iga->first] = iga->second;

	// 更新体单元索引信息
	VEindex vei;

	for (i = 1; i < numVEold + 1; i++)
	{
		vei.gid = VEgid[i];
		vei.Isin = 0;

		VEindexs.push_back(vei);
	}

	for (igaVE = gidVEs_add.begin(); igaVE != gidVEs_add.end(); ++igaVE)
	{
		vei.gid = igaVE->second;
		vei.Isin = 1;

		VEindexs.push_back(vei);
	}

	// =============================
	// 释放内存
	// =============================

	for (i = 0; i < num_r; i++)
	{
		free(r_data[i]);
		free(r_data_ves[i]);
	}

	if (num_s > 0)
	{
		free(dest);
		free(src);
		free(size);
		free(r_length);
		free(r_data_ves);
		free(s_length);
		free(s_data_ves);

		free(s_data);
		free(r_data);
	}

	MPI_Type_free(&mpivetype);

	free(oldgid);

	return newgid;
}

void Record_LWR_count(double LWR, int *count)
{ // record count of length_width_ratio in each interval
	if (LWR >= 1 && LWR < 1.5)
		count[0]++;
	if (LWR >= 1.5 && LWR < 2)
		count[1]++;
	if (LWR >= 2 && LWR < 3)
		count[2]++;
	if (LWR >= 3 && LWR < 4)
		count[3]++;
	if (LWR >= 4 && LWR < 5)
		count[4]++;
	if (LWR >= 5 && LWR < 6)
		count[5]++;
}

bool meshQualityEvaluation(void *mesh, int id, std::string OUTPUT_PATH)
{
	nglib::Ng_Mesh *newMesh = (nglib::Ng_Mesh *)mesh;
	int surfidx;
	int nse = nglib::Ng_GetNSE(newMesh);
	int np = nglib::Ng_GetNP(newMesh);
	int *surfpoints = new int[3];
	double xyz[3][3];
	double LWR;
	double LWR_sum = 0;
	double LWR_max = 0;
	double LWR_min = 0x3f3f3f;
	double JAC;
	double JAC_sum = 0;
	double JAC_max = 0;
	double JAC_min = 0x3f3f3f;
	double MinIA;
	double MaxIA;
	double IA_max = 0;
	double IA_min = 0x3f3f3f;
	double TRIS;
	double TRIS_sum = 0;
	double TRIS_max = 0;
	double TRIS_min = 0x3f3f3f;

	int Aspect_Ratio_count[6] = {0, 0, 0, 0, 0, 0}; // 记录长宽比

	for (int i = 0; i < nse; i++)
	{
		nglib::Ng_GetSurfaceElement(newMesh, i + 1, surfpoints, surfidx);
		for (int k = 0; k < 3; k++)
		{ // Each face has three points
			nglib::Ng_GetPoint((nglib::Ng_Mesh *)newMesh, surfpoints[k], xyz[k]);
		}
		LWR = length_width_ratio(xyz);

		Record_LWR_count(LWR, Aspect_Ratio_count);

		LWR_sum += LWR;
		LWR_min = (std::min)(LWR_min, LWR);
		LWR_max = (std::max)(LWR_max, LWR);
		// JAC = triangle_jacobian_ratio(xyz) ;
		// JAC_sum += JAC;
		// JAC_min = (std::min)(JAC_min,JAC) ;
		// JAC_max = (std::max)(JAC_max,JAC) ;
		MinIA = min_internal_angle(xyz);
		MaxIA = max_internal_angle(xyz);
		IA_min = (std::min)(IA_min, MinIA);
		IA_max = (std::max)(IA_max, MaxIA);
		TRIS = triangle_skew(xyz);
		TRIS_sum += TRIS;
		TRIS_min = (std::min)(TRIS_min, TRIS);
		TRIS_max = (std::max)(TRIS_max, TRIS);
	}
	int *Sum_Aspect_Ratio_count;
	if(id == 0)
		Sum_Aspect_Ratio_count = new int(6);
	for(int i = 0; i < 6; i++)
		MPI_Reduce(&Aspect_Ratio_count[i],&Sum_Aspect_Ratio_count[i], 1 , MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if(id == 0)
	{
		// printf("Sum_Aspect_Ratio_count : %d\n",Sum_Aspect_Ratio_count[0]);
		std::string savename = OUTPUT_PATH + "meshQuality/meshQuality.txt";
		FILE *fp = std::fopen(savename.c_str(), "w");
		int Sum_Count_Surface = 0;
		for(int i = 0; i < 6; i++)
			Sum_Count_Surface += Sum_Aspect_Ratio_count[i];
		// for(int i = 0; i < 6; i++)
		fprintf(fp, "Sum_Aspect_Ratio(1-1.5): %f \r\n", (float)Sum_Aspect_Ratio_count[0]/(float)Sum_Count_Surface);
		fprintf(fp, "Sum_Aspect_Ratio(1.5-2): %f \r\n", (float)Sum_Aspect_Ratio_count[1]/(float)Sum_Count_Surface);
		fprintf(fp, "Sum_Aspect_Ratio(2-3): %f \r\n", (float)Sum_Aspect_Ratio_count[2]/(float)Sum_Count_Surface);
		fprintf(fp, "Sum_Aspect_Ratio(3-4): %f \r\n", (float)Sum_Aspect_Ratio_count[3]/(float)Sum_Count_Surface);
		fprintf(fp, "Sum_Aspect_Ratio(4-5): %f \r\n", (float)Sum_Aspect_Ratio_count[4]/(float)Sum_Count_Surface);
		fprintf(fp, "Sum_Aspect_Ratio(5-6): %f \r\n", (float)Sum_Aspect_Ratio_count[5]/(float)Sum_Count_Surface);
		delete []Sum_Aspect_Ratio_count;
		std::fclose(fp);
	}
	int volidx;
	int ne = nglib::Ng_GetNE(newMesh);
	int *volpoints = new int[4];
	double Vxyz[4][3];
	double VLWR;
	double VLWR_sum = 0;
	double VLWR_max = 0;
	double VLWR_min = 0x3f3f3f;
	double VJAC;
	double VJAC_sum = 0;
	double VJAC_max = 0;
	double VJAC_min = 0x3f3f3f;
	double VMinIA;
	double VMaxIA;
	double VIA_max = 0;
	double VIA_min = 0x3f3f3f;
	double VTRIS;
	double VTRIS_sum = 0;
	double VTRIS_max = 0;
	double VTRIS_min = 0x3f3f3f;
	for (int i = 0; i < ne; i++)
	{
		nglib::Ng_GetVolumeElement(newMesh, i + 1, volpoints, volidx);
		for (int k = 0; k < 4; k++)
		{ // Each face has three points
			nglib::Ng_GetPoint((nglib::Ng_Mesh *)newMesh, volpoints[k], Vxyz[k]);
		}
		VLWR = tetrahedrons_length_width_ratio(Vxyz);
		VLWR_sum += VLWR;
		VLWR_min = (std::min)(VLWR_min, VLWR);
		VLWR_max = (std::max)(VLWR_max, VLWR);
		// VJAC = tetrahedrons_jacobian_ratio(Vxyz) ;
		// VJAC_sum += VJAC;
		// VJAC_min = (std::min)(VJAC_min,VJAC) ;
		// VJAC_max = (std::max)(VJAC_max,VJAC) ;
		VIA_min = (std::min)(VIA_min, VMinIA);
		VIA_max = (std::max)(VIA_max, VMaxIA);
		VTRIS = tetrahedrons_skew(Vxyz);
		VTRIS_sum += VTRIS;
		VTRIS_min = (std::min)(VTRIS_min, VTRIS);
		VTRIS_max = (std::max)(VTRIS_max, VTRIS);
		double face_xyz[4][3][3];
		int u[4][3] =
			{
				{0, 1, 2},
				{0, 1, 3},
				{0, 2, 3},
				{1, 2, 3},
			};
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 3; j++)
			{
				face_xyz[k][j][0] = Vxyz[u[k][j]][0];
				face_xyz[k][j][1] = Vxyz[u[k][j]][1];
				face_xyz[k][j][2] = Vxyz[u[k][j]][2];
			}
		double face_VMinIA[4];
		double face_VMaxIA[4];
		for (int k = 0; k < 4; k++)
		{
			face_VMinIA[k] = min_internal_angle(face_xyz[k]);
			face_VMaxIA[k] = max_internal_angle(face_xyz[k]);
		}
		VMinIA = (std::min)({face_VMinIA[0], face_VMinIA[1], face_VMinIA[2], face_VMinIA[3]});
		VMaxIA = (std::max)({face_VMaxIA[0], face_VMaxIA[1], face_VMaxIA[2], face_VMaxIA[3]});
	}

	std::string savename = OUTPUT_PATH + "meshQuality/meshQuality" + std::to_string(id) + ".txt";
	FILE *fp = std::fopen(savename.c_str(), "w");
	if (fp == NULL)
	{
		std::cout << "File " << savename.c_str() << "canot open" << std::endl;
	}
	else
	{

		fprintf(fp, "Point_Num: %d SurfEle_Num: %d SoildEle_Num: %d \r\n", np, nse, ne);
		fprintf(fp, "\r\n");
		fprintf(fp, "triangle_length_width_ratio_min: %f \r\n", LWR_min);
		fprintf(fp, "triangle_length_width_ratio_max: %f \r\n", LWR_max);
		fprintf(fp, "triangle_length_width_ratio_mean: %f \r\n", LWR_sum / nse);

		fprintf(fp, "Aspect_Ratio(1-1.5): %d \r\n", Aspect_Ratio_count[0]);
		fprintf(fp, "Aspect_Ratio(1.5-2): %d \r\n", Aspect_Ratio_count[1]);
		fprintf(fp, "Aspect_Ratio(2-3): %d \r\n", Aspect_Ratio_count[2]);
		fprintf(fp, "Aspect_Ratio(3-4): %d \r\n", Aspect_Ratio_count[3]);
		fprintf(fp, "Aspect_Ratio(4-5): %d \r\n", Aspect_Ratio_count[4]);
		fprintf(fp, "Aspect_Ratio(5-6): %d \r\n", Aspect_Ratio_count[5]);

		// fprintf(fp,"triangle_jacobian_ratio_min: %f \r\n",JAC_min);
		// fprintf(fp, "triangle_jacobian_ratio_max: %f \r\n",JAC_max);
		// fprintf(fp, "triangle_jacobian_ratio_mean: %f \r\n",JAC_sum / nse);
		fprintf(fp, "triangle_skew_min: %f \r\n", TRIS_min);
		fprintf(fp, "triangle_skew_max: %f \r\n", TRIS_max);
		fprintf(fp, "triangle_skew_mean: %f \r\n", TRIS_sum / nse);
		fprintf(fp, "triangle_internal_angle_min: %f \r\n", MinIA);
		fprintf(fp, "triangle_internal_angle_max: %f \r(n", MaxIA);
		fprintf(fp, "\r\n");
		fprintf(fp, "tetrahedrons_length_width_ratio_min: %f \r\n", VLWR_min);
		fprintf(fp, "tetrahedrons_length_width_ratio_max: %f \r\n", VLWR_max);
		fprintf(fp, "tetrahedrons_length_width_ratio_mean: %f \r\n", VLWR_sum / ne);
		// fprintf(fp, "tetrahedrons_jacobian_ratio_min: %f \r\n",VJAC_min);
		// fprintf(fp, "tetrahedrons_jacobian_ratio_max: %f \r\n",VJAC_max) ;
		// fprintf(fp,"tetrahedrons_jacobian_ratio_mean: %f \r\n",VJAC_sum / ne);
		fprintf(fp, "tetrahedrons_skew_min: %f \r\n", VTRIS_min);
		fprintf(fp, "tetrahedrons_skew_max: %f \r\n", VTRIS_max);
		fprintf(fp, "tetrahedrons_skew_mean: %f \r\n", VTRIS_sum / ne);
		fprintf(fp, "tetrahedrons_internal_angle_min: %f \r\n", VMinIA);
		fprintf(fp, "tetrahedrons_internal_angle_max: %f \r\n", VMaxIA);
	}
	std::fclose(fp);
	return false;
}

double triangle_jacobian_ratio(const double points[][3])
{
	double j11 = points[1][0] - points[0][0];
	double j12 = points[1][1] - points[0][1];
	double j13 = points[1][2] - points[0][2];
	double j21 = points[2][0] - points[0][0];
	double j22 = points[2][1] - points[0][1];
	double j23 = points[2][2] - points[0][2];
	double determin = j11 * j22 + j12 * j23 + j13 * j21 - j13 * j22 - j12 * j21 - j11 * j23;
	if (determin <= 0)
		determin = -determin;
	return determin;
}

double tetrahedrons_jacobian_ratio(const double points[][3])
{
	double j11 = points[1][0] - points[0][0];
	double j12 = points[1][1] - points[0][1];
	double j13 = points[1][2] - points[0][2];
	double j21 = points[2][0] - points[0][0];
	double j22 = points[2][1] - points[0][1];
	double j23 = points[2][2] - points[0][2];
	double j31 = points[3][0] - points[0][0];
	double j32 = points[3][1] - points[0][1];
	double j33 = points[3][2] - points[0][2];
	double determin = j11 * j22 * j33 + j12 * j23 * j31 + j13 * j21 * j32 - j13 * j22 * j31 - j12 * j21 * j33 - j11 * j23 * j32;
	if (determin <= 0)
		determin = -determin;
	return determin;
}

double getLenght(const double a1[], const double a2[])
{
	return sqrt((a1[0] - a2[0]) * (a1[0] - a2[0]) + (a1[1] - a2[1]) * (a1[1] - a2[1]) + (a1[2] - a2[2]) * (a1[2] - a2[2]));
}

//计算三个点构成的长度宽度比列
double length_width_ratio(const double points[][3])
{
	double MIN = 0x3f3f3f, MAX = 0;
	for (int i = 0; i < 3; i++)
		for (int j = i + 1; j < 3; j++)
		{
			MIN = (std::min)(MIN, getLenght(points[i], points[j]));
			MAX = (std::max)(MAX, getLenght(points[i], points[j]));
		}

	return MAX / MIN;
}

double tetrahedrons_length_width_ratio(const double points[][3])
{
	double MIN = 0x3f3f3f, MAX = 0;
	for (int i = 0; i < 4; i++)
		for (int j = i + 1; j < 4; j++)
		{
			MIN = (std::min)(MIN, getLenght(points[i], points[j]));
			MAX = (std::max)(MAX, getLenght(points[i], points[j]));
		}
	return MAX / MIN;
}

double min_internal_angle(const double points[][3])
{
	double l1 = getLenght(points[0], points[1]);
	double l2 = getLenght(points[0], points[2]);
	double l3 = getLenght(points[1], points[2]);
	if (l1 > l2)
		std::swap(l1, l2);
	if (l2 > l3)
		std::swap(l2, l3);
	if (l1 > l2)
		std::swap(l1, l2);
	double cos_a = (l2 * l2 + l3 * l3 - l1 * l1) / (2 * l2 * l3);
	return acos(cos_a) * 180 / M_PI;
}

double max_internal_angle(const double points[][3])
{
	double l1 = getLenght(points[0], points[1]);
	double l2 = getLenght(points[0], points[2]);
	double l3 = getLenght(points[1], points[2]);
	if (l1 > l2)
		std::swap(l1, l2);
	if (l2 > l3)
		std::swap(l2, l3);
	if (l1 > l2)
		std::swap(l1, l2);
	double cos_c = (l2 * l2 + l1 * l1 - l3 * l3) / (2 * l2 * l1);
	return acos(cos_c) * 180 / M_PI;
}

double func(const double a1[], const double a2[], const double a3[])
{
	double mid_12[3] = {(a1[0] + a2[0]) / 2, (a1[1] + a2[1]) / 2, (a1[2] + a2[2]) / 2};
	double mid_23[3] = {(a2[0] + a3[0]) / 2, (a2[1] + a3[1]) / 2, (a2[2] + a3[2]) / 2};
	double mid_13[3] = {(a1[0] + a3[0]) / 2, (a1[1] + a3[1]) / 2, (a1[2] + a3[2]) / 2};

	double l_1_mid23 = getLenght(a1, mid_23);

	double l_1_mid13 = getLenght(a1, mid_13);
	double l_mid23_mid13 = getLenght(mid_23, mid_13);

	double A = acos((l_1_mid23 * l_1_mid23 + l_mid23_mid13 * l_mid23_mid13 - l_1_mid13 * l_1_mid13) / (2 * l_1_mid23 * l_mid23_mid13)) * 180 / M_PI;

	double l_1_mid12 = getLenght(a1, mid_12);
	double l_mid23_mid12 = getLenght(mid_12, mid_23);

	double B = acos((l_1_mid23 * l_1_mid23 + l_mid23_mid12 * l_mid23_mid12 - l_1_mid12 * l_1_mid12) / (2 * l_1_mid23 * l_mid23_mid12)) * 180 / M_PI;

	return (std::min)(A, B);
}
//计算三角形的偏斜度（skew）。偏斜度定义为三个内角中最小角度与90度的差值。
double triangle_skew(const double points[][3])
{
	return 90 - (std::min)({func(points[0], points[1], points[2]), func(points[1], points[0], points[2]), func(points[2], points[1], points[0])});
}

double tetrahedrons_skew(double points[][3])
{
	double MIN = 0x3f3f3f;
	for (int i = 0; i < 4; i++)
	{
		MIN = (std::min)(MIN, triangle_skew(points));
		if (i != 3)
		{
			for (int j = 0; j < 3; j++)
			{
				std::swap(points[3][j], points[i][j]);
			}
		}
	}
	return MIN;
}
