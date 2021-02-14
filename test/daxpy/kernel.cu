#include <CL/cl.h>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <chrono>

const unsigned int SIZE = 204800000;
const int treadsNum = 2;
size_t localSize_gpu = 512;
size_t localSize_cpu = 512;


const char* daxpy_gpu = "__kernel void da_gpu (              \n"
"const int n,                                            \n"
"double a,                                               \n"
"__global double * x,                                    \n"
"const int incx,                                         \n"
"__global double * y,                                    \n"
"const int incy                                          \n"
") {                                                     \n"
"int ind = get_global_id(0);                             \n"
"if (ind < n)                                            \n"
"    y[ind * incy] = y[ind * incy] + a * x[ind * incx];  \n"
"}";


void daxpy(int n, double a, double* x, int incx, double* y, int incy) {
	for (int i = 0; i < n; i++) {
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

void daxpy_omp(int n, double a, double* x, int incx, double* y, int incy) {
#pragma omp parallel for 
	for (int i = 0; i < n; i++) {
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

template <typename T>
bool checkResults(T* first, T* second, int n) {
	double eps = 5e-3;
	for (int i = 0; i < n; i++) {
		if (fabs(first[i] - second[i]) > eps) {
			return false;
		}
	}
	return true;
}

int main() {
	std::cout << "SIZE = " << SIZE << std::endl;
	std::cout << "Thread num = " << treadsNum << std::endl;
	bool check_omp_daxpy;
	bool check_gpu_daxpy;
	bool check_cpu_daxpy;
	double* data_x_d = new double[SIZE];
	double* data_y_d = new double[SIZE];

	float a_d = (float)rand() / RAND_MAX;
	int incx = 1;
	int incy = 1;
	float tmp_x_d= 2.0f;//(float)rand() / RAND_MAX+7.333;
	float tmp_y_d = 3.5f;//(float)rand() / RAND_MAX+11.666;
	//Линейное время
	for (int i = 0; i < SIZE; i++) {
		data_x_d[i] = tmp_x_d;
		data_y_d[i] = tmp_y_d;
	}
	double start = omp_get_wtime();
	daxpy(SIZE, a_d, data_x_d, incx, data_y_d, incy);
	double finish = omp_get_wtime();
	double linear_time_daxpy = finish - start;
	//OMP
	double* data_x_fomp_d = new double[SIZE];
	double* data_y_fomp_d = new double[SIZE];
	for (int i = 0; i < SIZE; i++) {
		data_x_fomp_d[i] = tmp_x_d;
		data_y_fomp_d[i] = tmp_y_d;
	}
	start = omp_get_wtime();
	daxpy_omp(SIZE, a_d, data_x_fomp_d, incx, data_y_fomp_d, incy);
	finish = omp_get_wtime();
	double omp_time_daxpy = finish - start;
	check_omp_daxpy = checkResults(data_y_d, data_y_fomp_d, SIZE);

	delete[] data_x_fomp_d;
	delete[] data_y_fomp_d;



	std::cout << "++++++++++++++++++DAXPY+++++++++++++++++++" << std::endl;
	std::cout << "+-----------------Check------------------+" << std::endl;
	std::cout << "+-------OMP--------Cpu---------GPU-------+" << std::endl;
	std::cout << "+--------" << check_omp_daxpy << "----------" << 0 << "-----------" << 0 << "--------+" << std::endl;
	std::cout << " Linear time " << linear_time_daxpy * (1e+03) << "ms" << std::endl;
	std::cout << " Omp time " << omp_time_daxpy * (1e+03) << "ms" << std::endl;
//	std::cout << " GPU time1 " << (cl_double)(end_s1 - start_s1)*(cl_double)(1e-06) << "ms" << std::endl;
	return 0;
}