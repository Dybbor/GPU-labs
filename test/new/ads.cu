//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.h"
#include <omp.h>
#include <iostream>
#include <string>

void saxpy(const int& n, const float& a, const float* x, const int& incx, float* y, const int& incy) {
	for (int i = 0; i < n; i++) {
		y[i * incy] = a * x[i * incx] + y[i * incy];
	}
}
void daxpy(const int& n, const double& a, const double* x, const int& incx, double* y, const int& incy) {
	for (int i = 0; i < n; i++) {
		y[i * incy] = a * x[i * incx] + y[i * incy];
	}
}


void saxpy_omp(const int& n, const float& a, const float* x, const int& incx, float* y, const int& incy) {
#pragma omp parallel for 
	for (int i = 0; i < n; i++) {
		y[i * incy] = a * x[i * incx] + y[i * incy];
	}
}
void daxpy_omp(const int& n, const double& a, const double* x, const int& incx, double* y, const int& incy) {
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		y[i * incy] = a * x[i * incx] + y[i * incy];
	}
}

const char* saxpy_GPU =
"__kernel void saxpy_GPU(const int n, const float a, __global float* x, const int incx, __global float* y, const int incy) { \n"\
"int i = get_global_id(0);\n"\
"y[i * incy] = a * x[i * incx] + y[i * incy]; \n"\
"}";

const char* daxpy_GPU =
"__kernel void daxpy_GPU(const int n, const double a, __global double* x, const int incx, __global double* y, const int incy) { \n"\
"int i = get_global_id(0);\n"
"y[i * incy] = a * x[i * incx] + y[i * incy]; \n"
"}";

const int THREADS = 4;
const unsigned int SIZE = 10240000;

int main() {

	double start_time;
	double end_time;
	double linear_time_saxpy;
	double linear_time_daxpy;
	double omp_time_saxpy;
	double omp_time_daxpy;
	double gpu_time_saxpy;
	double gpu_time_daxpy;

	float* y_f = new float[SIZE];
	float* x_f = new float[SIZE];
	double* y_d = new double[SIZE];
	double* x_d = new double[SIZE];
//#pragma omp parallel
	printf("THREADS %d\n", omp_get_max_threads());
	for (int i = 0; i < SIZE; i++) {
		y_f[i] = 1.0f;
		x_f[i] = 1.0f;
		y_d[i] = 1;
		x_d[i] = 1;
	}

	//linear
	start_time = omp_get_wtime();
	saxpy(SIZE, 2.0f, x_f, 1, y_f, 1);
	end_time = omp_get_wtime();
	linear_time_saxpy = end_time - start_time;

	start_time = omp_get_wtime();
	daxpy(SIZE, 2, x_d, 1, y_d, 1);
	end_time = omp_get_wtime();
	linear_time_daxpy = end_time - start_time;



	//OpenMP
	omp_set_num_threads(THREADS);

	for (int i = 0; i < SIZE; i++) {
		y_f[i] = 1.0f;
		x_f[i] = 1.0f;
		y_d[i] = 1;
		x_d[i] = 1;
	}

	start_time = omp_get_wtime();
	saxpy_omp(SIZE, 2.0f, x_f, 1, y_f, 1);
	end_time = omp_get_wtime();
	omp_time_saxpy = end_time - start_time;

	start_time = omp_get_wtime();
	daxpy_omp(SIZE, 2, x_d, 1, y_d, 1);
	end_time = omp_get_wtime();
	omp_time_daxpy = end_time - start_time;


	//GPU

	for (int i = 0; i < SIZE; i++) {
		y_f[i] = 1.0f;
		x_f[i] = 1.0f;
		y_d[i] = 1;
		x_d[i] = 1;
	}

	cl_int error = 0;

	cl_uint num_platforms = 0;
	clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platform = NULL;
	std::cout << num_platforms << std::endl;
	if (0 < num_platforms) {
		cl_platform_id* platforms = new cl_platform_id[num_platforms];
		clGetPlatformIDs(num_platforms, platforms, NULL);
		platform = platforms[0];

		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
		std::cout << platform_name << std::endl;

		delete[] platforms;
	}

	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM,
										   (cl_context_properties)platform, 0 };

	cl_context context =
		clCreateContextFromType((NULL == platform) ? NULL : properties,
			CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed" << std::endl;
	}

	size_t size = 0;

	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);

	cl_device_id device = 0;
	if (size > 0) {
		cl_device_id* devices = (cl_device_id*)alloca(size);
		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
		device = devices[0];

		char device_name[128];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed" << std::endl;
	}




	//PROGRAMS
	size_t srclen1[] = { strlen(saxpy_GPU) };

	cl_program program1 = clCreateProgramWithSource(context, 1, &saxpy_GPU, srclen1, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateProgramWithSource" << std::endl;

	error = clBuildProgram(program1, 1, &device, NULL, NULL, NULL);
	//if (error == CL_SUCCESS) std::cout << "Build1: true" << std::endl;
	//else std::cout << "Build1 error: " << error << std::endl;

	cl_kernel kernel1 = clCreateKernel(program1, "saxpy_GPU", &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateKernel  " << error << std::endl;



	size_t srclen2[] = { strlen(daxpy_GPU) };

	cl_program program2 = clCreateProgramWithSource(context, 1, &daxpy_GPU, srclen2, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateProgramWithSource" << std::endl;

	error = clBuildProgram(program2, 1, &device, NULL, NULL, NULL);
	//if (error == CL_SUCCESS) std::cout << "Build2: true" << std::endl;
	//else std::cout << "Build2 error: " << error << std::endl;

	cl_kernel kernel2 = clCreateKernel(program2, "daxpy_GPU", &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateKernel  " << error << std::endl;




	cl_mem input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * SIZE, NULL, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateBuffer1" << std::endl;
	cl_mem output1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * SIZE, NULL, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateBuffer2" << std::endl;

	cl_mem input2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIZE, NULL, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateBuffer1" << std::endl;
	cl_mem output2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * SIZE, NULL, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateBuffer2" << std::endl;

	error = clEnqueueWriteBuffer(queue, input1, CL_TRUE, 0, sizeof(float) * SIZE, x_f, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clEnqueueWriteBuffer1 " << error << std::endl;

	error = clEnqueueWriteBuffer(queue, input2, CL_TRUE, 0, sizeof(double) * SIZE, x_d, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clEnqueueWriteBuffer2 " << error << std::endl;

	error = clEnqueueWriteBuffer(queue, output1, CL_TRUE, 0, sizeof(float) * SIZE, y_f, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clEnqueueWriteBuffer1 " << error << std::endl;

	error = clEnqueueWriteBuffer(queue, output2, CL_TRUE, 0, sizeof(double) * SIZE, y_d, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clEnqueueWriteBuffer2 " << error << std::endl;

	int count = SIZE;
	float a1 = 2.0f;
	double a2 = 2;
	int incx = 1;
	int incy = 1;

	error = clSetKernelArg(kernel1, 0, sizeof(int), &count);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg0" << std::endl;

	error = clSetKernelArg(kernel1, 1, sizeof(float), &a1);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg1" << std::endl;

	error = clSetKernelArg(kernel1, 2, sizeof(cl_mem), &input1);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg2" << std::endl;
	//std::cout << "HERE" << std::endl;
	error = clSetKernelArg(kernel1, 3, sizeof(int), &incx);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg3" << std::endl;
	error = clSetKernelArg(kernel1, 4, sizeof(cl_mem), &output1);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg4" << std::endl;
	error = clSetKernelArg(kernel1, 5, sizeof(int), &incy);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg5" << std::endl;


	error = clSetKernelArg(kernel2, 0, sizeof(int), &count);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg0" << std::endl;
	error = clSetKernelArg(kernel2, 1, sizeof(double), &a2);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg1" << std::endl;
	error = clSetKernelArg(kernel2, 2, sizeof(cl_mem), &input2);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg2" << std::endl;
	error = clSetKernelArg(kernel2, 3, sizeof(int), &incx);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg3" << std::endl;
	error = clSetKernelArg(kernel2, 4, sizeof(cl_mem), &output2);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg4" << std::endl;
	error = clSetKernelArg(kernel2, 5, sizeof(int), &incy);
	if (error != CL_SUCCESS) std::cout << "Error clSetKernelArg5" << std::endl;



	//----------------------------------
	size_t group = 256;
	std::cout << "Size: " << count << ". Group: " << group << ". Groups: " << count / group << std::endl;

	size_t count1 = SIZE;

	start_time = omp_get_wtime();
	error = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL, &count1, &group, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clEnqueueNDRangeKernel:" << error << std::endl;

	error = clFinish(queue);
	if (error != CL_SUCCESS) std::cout << "Error clFinish" << std::endl;
	end_time = omp_get_wtime();
	gpu_time_saxpy = end_time - start_time;

	start_time = omp_get_wtime();
	error = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, &count1, &group, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clEnqueueNDRangeKernel:" << error << std::endl;
	error = clFinish(queue);
	if (error != CL_SUCCESS) std::cout << "Error clFinish" << std::endl;
	end_time = omp_get_wtime();
	gpu_time_daxpy = end_time - start_time;

	error = clEnqueueReadBuffer(queue, output1, CL_TRUE, 0, sizeof(float) * count, y_f, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clEnqueueReadBuffer1 " << error << std::endl;

	error = clEnqueueReadBuffer(queue, output2, CL_TRUE, 0, sizeof(float) * count, y_d, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clEnqueueReadBuffer2" << error << std::endl;

	std::cout << "-------------------GPU-------------------" << std::endl;
	std::cout << "SAXPY time: " << gpu_time_saxpy << std::endl;
	std::cout << "DAXPY time: " << gpu_time_daxpy << std::endl;

	std::cout << std::endl;

	std::cout << "-------------------LINEAR-------------------" << std::endl;
	std::cout << "SAXPY time: " << linear_time_saxpy << std::endl;
	std::cout << "DAXPY time: " << linear_time_daxpy << std::endl;

	std::cout << std::endl;

	std::cout << "-------------------OpenMP-------------------" << std::endl;
	std::cout << "SAXPY time: " << omp_time_saxpy << std::endl;
	std::cout << "DAXPY time: " << omp_time_daxpy << std::endl;


	clReleaseMemObject(input1);
	clReleaseMemObject(output1);
	clReleaseMemObject(input2);
	clReleaseMemObject(output2);
	clReleaseProgram(program1);
	clReleaseProgram(program2);
	clReleaseKernel(kernel1);
	clReleaseKernel(kernel2);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}