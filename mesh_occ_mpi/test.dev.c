#include <stdint.h>
#include <float.h>
#include <math.h>
#include <compiler/m3000.h>
#include "hthread_device.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 已有的几何函数
void swap_double(double a, double b) {
    double temp = a;
    a = b;
    b = temp;
}

static inline
double getLenght(const double a1[], const double a2[])
{
	return sqrt((a1[0] - a2[0]) * (a1[0] - a2[0]) + (a1[1] - a2[1]) * (a1[1] - a2[1]) + (a1[2] - a2[2]) * (a1[2] - a2[2]));
}

static inline
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

	return (A < B)? A:B;
}

//计算三角形的偏斜度（skew）。偏斜度定义为三个内角中最小角度与90度的差值。
static inline
double triangle_skew(const double points[][3])
{
    double angle1 = func(points[0], points[1], points[2]);
    double angle2 = func(points[1], points[0], points[2]);
    double angle3 = func(points[2], points[1], points[0]);

    double min_angle = angle1;
    if (angle2 < min_angle) min_angle = angle2;
    if (angle3 < min_angle) min_angle = angle3;

    return 90.0 - min_angle; 
}

static inline
double tetrahedrons_length_width_ratio(const double points[][3])
{
	double MIN = 0x3f3f3f, MAX = 0;
	for (int i = 0; i < 4; i++)
		for (int j = i + 1; j < 4; j++)
		{
		    double len = getLenght(points[i], points[j]);
            if (len < MIN) MIN = len;
            if (len > MAX) MAX = len;
		}
	return MAX / MIN;
}

static inline
double tetrahedrons_skew(double points[][3])
{
	double MIN = 0x3f3f3f;
	for (int i = 0; i < 4; i++)
	{
        double len = triangle_skew(points);
        if (len < MIN) MIN = len;
		if (i != 3)
		{
			for (int j = 0; j < 3; j++)
			{
				swap_double(points[3][j], points[i][j]);
			}
		}
	}
	return MIN;
}

static inline
double min_internal_angle(const double points[][3])
{
	double l1 = getLenght(points[0], points[1]);
	double l2 = getLenght(points[0], points[2]);
	double l3 = getLenght(points[1], points[2]);
	if (l1 > l2)
		swap_double(l1, l2);
	if (l2 > l3)
		swap_double(l2, l3);
	if (l1 > l2)
		swap_double(l1, l2);
	double cos_a = (l2 * l2 + l3 * l3 - l1 * l1) / (2 * l2 * l3);
	return acos(cos_a) * 180 / M_PI;
}

static inline
double max_internal_angle(const double points[][3])
{
	double l1 = getLenght(points[0], points[1]);
	double l2 = getLenght(points[0], points[2]);
	double l3 = getLenght(points[1], points[2]);
	if (l1 > l2)
		swap_double(l1, l2);
	if (l2 > l3)
		swap_double(l2, l3);
	if (l1 > l2)
		swap_double(l1, l2);
	double cos_c = (l2 * l2 + l1 * l1 - l3 * l3) / (2 * l2 * l1);
	return acos(cos_c) * 180 / M_PI;
}
typedef struct {
  double VLWR_sum, VLWR_min, VLWR_max;
  double VTRIS_sum, VTRIS_min, VTRIS_max;
  double VIA_min,  VIA_max;
} QualityOut;


__global__
void mesh_quality_kernel(uint64_t node_id, uint64_t cluster_id, uint64_t ne,  double *Data, QualityOut *d_out )
{
    // 获取线程 ID 和线程总数
    uint64_t thread_id = get_thread_id();
    uint64_t threads_count = get_group_size();

    const uint64_t chunk = ne / threads_count;
    const uint64_t start = thread_id * chunk;
    const uint64_t end   = (thread_id == threads_count-1) ? ne : (start + chunk);

    double VLWR_sum=0.0, VLWR_min= 0x3f3f3f, VLWR_max=0;
    double VTRIS_sum=0.0, VTRIS_min=0x3f3f3f, VTRIS_max=0;
    double VIA_min=0x3f3f3f, VIA_max=0;

    const int face_idx[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};

    hthread_printf("[node %lu][clustr_id %lu][DSP %d] [dsp_thread_id %lu] [start: %lu - end: %lu] [chunk %lu] [threads_count %lu]\n",
                    node_id, cluster_id, get_core_id(), thread_id, start, end, chunk, threads_count);

    for (uint64_t i = start; i < end; ++i) {
        const double* base = Data + i*12;
        double Vxyz[4][3] = {
            {base[0], base[1], base[2]},
            {base[3], base[4], base[5]},
            {base[6], base[7], base[8]},
            {base[9], base[10], base[11]}
        };
/*         // --- Start: Added Print Logic ---
        if (node_id == 0 && cluster_id == 0) {
                 hthread_printf("[N0 C0 DSP%d Thr%lu] Element %lu Vxyz:\n", get_core_id(), thread_id, i);
                 for (int vert = 0; vert < 4; ++vert) {
                     hthread_printf("  Vert %d: (%f, %f, %f)\n", vert, Vxyz[vert][0], Vxyz[vert][1], Vxyz[vert][2]);
                 }
        }
        // --- End: Added Print Logic --- */

        const double VLWR = tetrahedrons_length_width_ratio(Vxyz);
        VLWR_sum += VLWR;
        if (VLWR < VLWR_min) VLWR_min = VLWR;
        if (VLWR > VLWR_max) VLWR_max = VLWR;

        const double VTRIS = tetrahedrons_skew(Vxyz);
        VTRIS_sum += VTRIS;
        if (VTRIS < VTRIS_min) VTRIS_min = VTRIS;
        if (VTRIS > VTRIS_max) VTRIS_max = VTRIS;

        // 面角
        double elem_min = DBL_MAX, elem_max = -DBL_MAX;
        double face_xyz[3][3];
        for (int f=0; f<4; ++f) {
            face_xyz[0][0]=Vxyz[face_idx[f][0]][0];
            face_xyz[0][1]=Vxyz[face_idx[f][0]][1];
            face_xyz[0][2]=Vxyz[face_idx[f][0]][2];
            face_xyz[1][0]=Vxyz[face_idx[f][1]][0];
            face_xyz[1][1]=Vxyz[face_idx[f][1]][1];
            face_xyz[1][2]=Vxyz[face_idx[f][1]][2];
            face_xyz[2][0]=Vxyz[face_idx[f][2]][0];
            face_xyz[2][1]=Vxyz[face_idx[f][2]][1];
            face_xyz[2][2]=Vxyz[face_idx[f][2]][2];

            const double min_a = min_internal_angle(face_xyz);
            const double max_a = max_internal_angle(face_xyz);
            if (min_a < elem_min) elem_min = min_a;
            if (max_a > elem_max) elem_max = max_a;
        }
        if (elem_min < VIA_min) VIA_min = elem_min;
        if (elem_max > VIA_max) VIA_max = elem_max;
    }

    hthread_printf("VLWR_sum:%f\n",VLWR_sum);
    hthread_printf("VLWR_min:%f\n",VLWR_min);
    hthread_printf("VLWR_max:%f\n",VLWR_max);
    hthread_printf("VTRIS_sum:%f\n",VTRIS_sum);
    hthread_printf("VTRIS_min:%f\n",VTRIS_min);
    hthread_printf("VTRIS_max:%f\n",VTRIS_max);
    hthread_printf("VIA_min:%f\n",VIA_min);
    hthread_printf("VIA_max:%f\n",VIA_max);
    // 写回“按线程槽位”的部分结果
    d_out[thread_id].VLWR_sum = VLWR_sum;
    d_out[thread_id].VLWR_min = VLWR_min;
    d_out[thread_id].VLWR_max = VLWR_max;

    d_out[thread_id].VTRIS_sum = VTRIS_sum;
    d_out[thread_id].VTRIS_min = VTRIS_min;
    d_out[thread_id].VTRIS_max = VTRIS_max;

    d_out[thread_id].VIA_min = VIA_min;
    d_out[thread_id].VIA_max = VIA_max;
}