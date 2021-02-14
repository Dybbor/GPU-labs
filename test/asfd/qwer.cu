#include <CL/cl.h>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <chrono>

const unsigned int SIZE = 51200000;
const int treadsNum = 4;
size_t localSize_gpu = 256;
size_t localSize_cpu = 256;


const char* saxpy_gpu = "__kernel void sa_gpu (              \n"
"const int n,                                            \n"
"float a,                                                \n"
"__global float * x,                                     \n"
"const int incx,                                         \n"
"__global float * y,                                     \n"
"const int incy                                          \n"
") {                                                     \n"
"int ind = get_global_id(0);                             \n"
"if (ind < n)                                            \n"
"    y[ind * incy] = y[ind * incy] + a * x[ind * incx];  \n"
"}";

const char* daxpy_gpu = "__kernel void da_gpu (              \n"
"const int n,                                            \n"
"double a,                                               \n"
"__global double * x,                                    \n"
"const int incx,                                         \n"
"__global double * y,                                    \n"
"const int incy                                          \n"
") {                                                     \n"
"int ind = get_global_id(0);                             \n"
"if (ind < n)   {                                         \n"
"    y[ind * incy] = y[ind * incy] + a * x[ind * incx];  \n"
"}}";

void saxpy(int n, float a, float* x, int incx, float* y, int incy) {
	for (int i = 0; i < n; i++) {
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

void daxpy(int n, double a, double* x, int incx, double* y, int incy) {
	for (int i = 0; i < n; i++) {
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

void saxpy_omp(int n, float a, float* x, int incx, float* y, int incy) {
#pragma omp parallel for
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
//int main() {
//	return 0;
//}
void main() {
	std::cout << "SIZE = " << SIZE << std::endl;
	std::cout << "Thread num = " << treadsNum << std::endl;

	double start_time;
	double end_time;
	double linear_time_saxpy;
	double linear_time_daxpy;
	double omp_time_saxpy;
	double omp_time_daxpy;
	double gpu_time_saxpy;
	double gpu_time_daxpy;
	double cpu_time_saxpy;
	double cpu_time_daxpy;
	bool check_omp_saxpy;
	bool check_gpu_saxpy;
	bool check_cpu_saxpy;
	bool check_omp_daxpy;
	bool check_gpu_daxpy;
	bool check_cpu_daxpy;
	float* data_x_f = new float[SIZE];
	float* data_x_fgpu_f;
	float* data_x_fomp_f;
	float* data_x_fcpu_f;
	float* data_y_f= new float[SIZE];
	float* data_y_fgpu_f;
	float* data_y_fomp_f;
	float* data_y_fcpu_f;
	float* results_fgpu_f;
	float* results_fcpu_f;


	float a = (float)rand() / RAND_MAX;
	int incx = 1;
	int incy = 1;
	float tmp_x = 2.0f;//(float)rand() / RAND_MAX+7.333;
	float tmp_y = 3.5f;//(float)rand() / RAND_MAX+11.666;
	//Линейное время
	for (int i = 0; i < SIZE; i++) {
		data_x_f[i] = tmp_x;
		data_y_f[i] = tmp_y;
	}
	double start = omp_get_wtime();
	saxpy(SIZE, a, data_x_f, incx, data_y_f, incy);
	double finish = omp_get_wtime();
	linear_time_saxpy = finish - start;
	//omp
	data_x_fomp_f = new float[SIZE];
	data_y_fomp_f = new float[SIZE];
	for (int i = 0; i < SIZE; i++) {
		data_x_fomp_f[i] = tmp_x;
		data_y_fomp_f[i] = tmp_y;
	}
	start = omp_get_wtime();
	saxpy_omp(SIZE, a, data_x_fomp_f, incx, data_y_fomp_f, incy);
	finish = omp_get_wtime();
	omp_time_saxpy = finish - start;
	check_omp_saxpy = checkResults(data_y_f, data_y_fomp_f, SIZE);
	delete[]data_x_fomp_f;
	delete[]data_y_fomp_f;
	//GPU
	data_x_fgpu_f = new float[SIZE];
	data_y_fgpu_f = new float[SIZE];
	for (int i = 0; i < SIZE; i++) {
		data_x_fgpu_f[i] = tmp_x;
		data_y_fgpu_f[i] = tmp_y;
	}
	cl_int error = 0;
	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	cl_platform_id platform = NULL;

	if (0 < numPlatforms) {
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);

		delete[] platforms;
	}

	cl_context_properties properties[3] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	cl_context context = clCreateContextFromType((NULL == platform) ? NULL : properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) { std::cout << "Create context from type failed: " << error << std::endl; }
	size_t size = 0;

	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
	cl_device_id device_id;
	if (size > 0) {
		cl_device_id * devices = new cl_device_id [size];
		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
		device_id = devices[0];

		char device_name[128];
		clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
		delete[] devices; 
	}
	

	if (error != CL_SUCCESS) { std::cout << "Create context failed: " << error << std::endl; }
	cl_command_queue queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
	if (error != CL_SUCCESS) { std::cout << "Create command queue with properties failed: " << error << std::endl; }
	size_t srclen[] = { strlen(saxpy_gpu) };
	cl_program program = clCreateProgramWithSource(context, 1, &saxpy_gpu, srclen, &error);
	if (error != CL_SUCCESS) { std::cout << "Create program failed: " << error << std::endl; }
	error = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Build program failed: " << error << std::endl; }
	cl_kernel kernel = clCreateKernel(program, "sa_gpu", &error);
	if (error != CL_SUCCESS) { std::cout << "Create kernel failed: " << error << std::endl; }
	size_t group = 0;
	size_t n = SIZE;
	cl_mem x = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * SIZE, NULL, NULL);
	cl_mem y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * SIZE, NULL, NULL);

	error = clEnqueueWriteBuffer(queue, x, CL_TRUE, 0, sizeof(float) * SIZE, data_x_f, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Enqueue write buffer data_x failed: " << error << std::endl; }
	error = clEnqueueWriteBuffer(queue, y, CL_TRUE, 0, sizeof(float) * SIZE, data_y_f, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Enqueue write buffer data_x failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 0, sizeof(int), &n);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for n failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 1, sizeof(float), &a);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for a failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for x failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 3, sizeof(int), &incx);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for incx failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for y failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 5, sizeof(int), &incy);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for incy failed: " << error << std::endl; }
	clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);

	cl_event evt;
	start = omp_get_wtime();
	error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &n, &localSize_gpu, 0, NULL, &evt);
	finish = omp_get_wtime();
	clWaitForEvents(1, &evt);

	cl_ulong start_s = 0, end_s = 0;
	error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_s, nullptr);
	if (error != CL_SUCCESS) {std::cout << "Error getting start time: " << error << std::endl;}
	error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_s, nullptr);
	if (error != CL_SUCCESS) {		std::cout << "Error getting end time: " << error << std::endl;}
	results_fgpu_f = new float[SIZE];
	clEnqueueReadBuffer(queue, y, CL_TRUE, 0, sizeof(float) * n, results_fgpu_f, 0, NULL, NULL);
	clFinish(queue);
	gpu_time_saxpy = finish - start;
	check_gpu_saxpy = checkResults(data_y_f, results_fgpu_f, SIZE);

	delete[] data_x_fgpu_f;
	delete[] data_y_fgpu_f;	
	delete[] results_fgpu_f;
	clReleaseMemObject(x);
	clReleaseMemObject(y);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	//CPU
	data_x_fcpu_f = new float[SIZE];
	data_y_fcpu_f = new float[SIZE];
	for (int i = 0; i < SIZE; i++) {
		data_x_fcpu_f[i] = tmp_x;
		data_y_fcpu_f[i] = tmp_y;
	}
	error = 0;
	numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	 platform = NULL;

	if (0 < numPlatforms) {
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[1];
		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
		delete[] platforms;
	}
	
	cl_context_properties propertiesn[3] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	context = clCreateContextFromType((NULL == platform) ? NULL : propertiesn, CL_DEVICE_TYPE_CPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) { std::cout << "Create context from type failed: " << error << std::endl; }
	size = 0;
	
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
	//device_id;
	if (size > 0) {
		cl_device_id * devices = new cl_device_id[size];
		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
		device_id = devices[0];

		char device_name[128];
		clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
		delete[] devices;
	}
	

	if (error != CL_SUCCESS) { std::cout << "Create context failed: " << error << std::endl; }
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
	if (error != CL_SUCCESS) { std::cout << "Create command queue with properties failed: " << error << std::endl; }
	size_t srclen1[] = { strlen(saxpy_gpu) };
	program = clCreateProgramWithSource(context, 1, &saxpy_gpu, srclen1, &error);
	if (error != CL_SUCCESS) { std::cout << "Create program failed: " << error << std::endl; }
	error = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Build program failed: " << error << std::endl; }
	kernel = clCreateKernel(program, "sa_gpu", &error);
	if (error != CL_SUCCESS) { std::cout << "Create kernel failed: " << error << std::endl; }

	group = 0;
	n = SIZE;
	x = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * SIZE, NULL, NULL);
	y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * SIZE, NULL, NULL);
	error = clEnqueueWriteBuffer(queue, x, CL_TRUE, 0, sizeof(float) * SIZE, data_x_fcpu_f, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Enqueue write buffer data_x failed: " << error << std::endl; }
	error = clEnqueueWriteBuffer(queue, y, CL_TRUE, 0, sizeof(float) * SIZE, data_y_fcpu_f, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Enqueue write buffer data_x failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 0, sizeof(int), &n);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for n failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 1, sizeof(float), &a);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for a failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for x failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 3, sizeof(int), &incx);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for incx failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for y failed: " << error << std::endl; }
	error = clSetKernelArg(kernel, 5, sizeof(int), &incy);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for incy failed: " << error << std::endl; }
	clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
	
	cl_event evt1;
	start = omp_get_wtime();
	error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &n, &localSize_cpu, 0, NULL, &evt1);
	finish = omp_get_wtime();
	clWaitForEvents(1, &evt1);
	cl_ulong start_so = 0, end_so = 0;
	error = clGetEventProfilingInfo(evt1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_so, nullptr);
	if (error != CL_SUCCESS) { std::cout << "Error getting start time: " << error << std::endl; }
	error = clGetEventProfilingInfo(evt1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_so, nullptr);
	if (error != CL_SUCCESS) { std::cout << "Error getting end time: " << error << std::endl; }
	results_fcpu_f = new float[SIZE];
	
	clEnqueueReadBuffer(queue, y, CL_TRUE, 0, sizeof(float) * n, results_fcpu_f, 0, NULL, NULL);
	clFinish(queue);
	cpu_time_saxpy = finish - start;
	check_cpu_saxpy = checkResults(data_y_f, results_fcpu_f, SIZE);
	
	delete[] data_x_fcpu_f;
	delete[] data_y_fcpu_f;
	delete[] results_fcpu_f;
	clReleaseMemObject(x);
	clReleaseMemObject(y);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);



	std::cout << "++++++++++++++++++SAXPY+++++++++++++++++++" << std::endl;
	std::cout << "+-----------------Check------------------+" << std::endl;
	std::cout << "+-------OMP--------Cpu---------GPU-------+" << std::endl;
	std::cout << "+--------" << check_omp_saxpy << "----------" << check_gpu_saxpy << "-----------" << check_cpu_saxpy << "--------+" << std::endl;
	std::cout << " Linear time " << linear_time_saxpy * (1e+03) <<"ms"<< std::endl;
	std::cout << " Omp time " << omp_time_saxpy * (1e+03) << "ms" << std::endl;
	//std::cout << " GPU time " << gpu_time_saxpy * (1e+03) << "ms" << std::endl;
	std::cout << " GPU time1 " << (cl_double)(end_s-start_s)*(cl_double)(1e-06) << "ms" << std::endl;
	std::cout << " CPU time " << (cl_double)(end_so-start_so)*(cl_double)(1e-06) << "ms" << std::endl;
	
	delete[] data_x_f;
	delete[] data_y_f;

/////////////daxpy
	double* data_x_d = new double[SIZE];
	double* data_y_d = new double[SIZE];
	double a_d = (double)rand() / RAND_MAX;
	incx = 1;
	incy = 1;
	double tmp_x_d = 2.0;
	double tmp_y_d = 3.5;
	//Линейное время
	for (int i = 0; i < SIZE; i++) {
		data_x_d[i] = tmp_x_d;
		data_y_d[i] = tmp_y_d;
	}
	start = omp_get_wtime();
	daxpy(SIZE, a_d, data_x_d, incx, data_y_d, incy);
	finish = omp_get_wtime();
	linear_time_daxpy = finish - start;
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
	omp_time_daxpy = finish - start;
	check_omp_daxpy = checkResults(data_y_d, data_y_fomp_d, SIZE);
	delete[] data_x_fomp_d;
	delete[] data_y_fomp_d;
	//Opencl GPU
	double* data_x_fgpu_d = new double[SIZE];
	double* data_y_fgpu_d = new double[SIZE];
	for (int i = 0; i < SIZE; i++) {
		data_x_fgpu_d[i] = tmp_x_d;
		data_y_fgpu_d[i] = tmp_y_d;
	}
	error = 0;
	numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	platform = NULL;

	if (0 < numPlatforms) {
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);

		delete[] platforms;
	}

	cl_context_properties properties_d[3] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	cl_context context_new = clCreateContextFromType((NULL == platform) ? NULL : properties_d, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) { std::cout << "Create context from type failed: " << error << std::endl; }
	size = 0;

	clGetContextInfo(context_new, CL_CONTEXT_DEVICES, 0, NULL, &size);
	device_id;
	if (size > 0) {
		cl_device_id * devices = new cl_device_id[size];
		clGetContextInfo(context_new, CL_CONTEXT_DEVICES, size, devices, NULL);
		device_id = devices[0];

		char device_name[128];
		clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, device_name, nullptr);
		//std::cout << device_name << std::endl;
		delete[] devices;
	}


	if (error != CL_SUCCESS) { std::cout << "Create context failed: " << error << std::endl; }
	cl_command_queue queue_new = clCreateCommandQueue(context_new, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
	if (error != CL_SUCCESS) { std::cout << "Create command queue with properties failed: " << error << std::endl; }
	size_t srclen2[] = { strlen(daxpy_gpu) };
	cl_program program_new = clCreateProgramWithSource(context_new, 1, &daxpy_gpu, srclen2, &error);
	if (error != CL_SUCCESS) { std::cout << "Create program failed: " << error << std::endl; }
	error = clBuildProgram(program_new, 1, &device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Build program failed: " << error << std::endl; }
	cl_kernel kernel_new = clCreateKernel(program_new, "da_gpu", &error);
	if (error != CL_SUCCESS) { std::cout << "Create kernel failed: " << error << std::endl; }
	group = 0;
	n = SIZE;
	cl_mem x_d = clCreateBuffer(context_new, CL_MEM_READ_ONLY, sizeof(double) * SIZE, NULL, NULL);
	cl_mem y_d = clCreateBuffer(context_new, CL_MEM_READ_WRITE, sizeof(double) * SIZE, NULL, NULL);
	error = clEnqueueWriteBuffer(queue_new, x_d, CL_TRUE, 0, sizeof(double) * SIZE, data_x_fgpu_d, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Enqueue write buffer data_x_fgpu_d failed: " << error << std::endl; }
	error = clEnqueueWriteBuffer(queue_new, y_d, CL_TRUE, 0, sizeof(double) * SIZE, data_y_fgpu_d, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Enqueue write buffer data_y_fgpu_d!!! failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 0, sizeof(int), &n);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for n failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 1, sizeof(double), &a_d);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for a failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 2, sizeof(cl_mem), &x_d);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for x failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 3, sizeof(int), &incx);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for incx failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 4, sizeof(cl_mem), &y_d);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for y failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 5, sizeof(int), &incy);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for incy failed: " << error << std::endl; }
	clGetKernelWorkGroupInfo(kernel_new, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);

	cl_event evt2;
	error = clEnqueueNDRangeKernel(queue_new, kernel_new, 1, NULL, &n, &localSize_gpu, 0, NULL, &evt2);
	clWaitForEvents(1, &evt2);

	cl_ulong start_s1 = 0, end_s1 = 0;
	error = clGetEventProfilingInfo(evt2, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_s1, nullptr);
	if (error != CL_SUCCESS) { std::cout << "Error getting start time here!: " << error << std::endl; }
	error = clGetEventProfilingInfo(evt2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_s1, nullptr);
	if (error != CL_SUCCESS) { std::cout << "Error getting end time: " << error << std::endl; }
	double* results_fgpu_d = new double[SIZE];
	error = clEnqueueReadBuffer(queue_new, y_d, CL_TRUE, 0, sizeof(double) * n, results_fgpu_d, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Error readBuffer: " << error << std::endl; }
	error = clFinish(queue_new);
	if (error != CL_SUCCESS) { std::cout << "Error clfinish: " << error << std::endl; }
	gpu_time_daxpy = end_s1 - start_s1;
	check_gpu_daxpy = checkResults(data_y_d, results_fgpu_d, SIZE);
	delete[] data_x_fgpu_d;
	delete[] data_y_fgpu_d;
	delete[] results_fgpu_d;
	error = clReleaseMemObject(x_d);
	if (error != CL_SUCCESS) { std::cout << "x_d " << error << std::endl; }
	error = clReleaseMemObject(y_d);
	if (error != CL_SUCCESS) { std::cout << "y_d " << error << std::endl; }
	error = clReleaseProgram(program_new);
	if (error != CL_SUCCESS) { std::cout << "program_new " << error << std::endl; }
	error = clReleaseKernel(kernel_new);
	if (error != CL_SUCCESS) { std::cout << "release kernel " << error << std::endl; }
	error = clReleaseCommandQueue(queue_new);
	if (error != CL_SUCCESS) { std::cout << "release queue " << error << std::endl; }
	error = clReleaseContext(context_new);
	if (error != CL_SUCCESS) { std::cout << "release context " << error << std::endl; }
	
	//CPU
	double* data_x_fcpu_d = new double[SIZE];
	double* data_y_fcpu_d = new double[SIZE];
	for (int i = 0; i < SIZE; i++) {
		data_x_fcpu_d[i] = tmp_x_d;
		data_y_fcpu_d[i] = tmp_y_d;
	}
	error = 0;
	numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	platform = NULL;

	if (0 < numPlatforms) {
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[1];
		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);

		delete[] platforms;
	}

	cl_context_properties  properties_d1[3] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	
	context_new = clCreateContextFromType((NULL == platform) ? NULL : properties_d1, CL_DEVICE_TYPE_CPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) { std::cout << "Create context from type failed: " << error << std::endl; }
	size = 0;

	clGetContextInfo(context_new, CL_CONTEXT_DEVICES, 0, NULL, &size);
	device_id;
	if (size > 0) {
		cl_device_id * devices = new cl_device_id[size];
		clGetContextInfo(context_new, CL_CONTEXT_DEVICES, size, devices, NULL);
		device_id = devices[0];

		char device_name[128];
		clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, device_name, nullptr);
		//std::cout << device_name << std::endl;
		delete[] devices;
	}

	if (error != CL_SUCCESS) { std::cout << "Create context failed: " << error << std::endl; }
	queue_new = clCreateCommandQueue(context_new, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
	if (error != CL_SUCCESS) { std::cout << "Create command queue with properties failed: " << error << std::endl; }
	size_t srclen21[] = { strlen(daxpy_gpu) };
	program_new = clCreateProgramWithSource(context_new, 1, &daxpy_gpu, srclen21, &error);
	if (error != CL_SUCCESS) { std::cout << "Create program failed: " << error << std::endl; }
	error = clBuildProgram(program_new, 1, &device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Build program failed: " << error << std::endl; }
	kernel_new = clCreateKernel(program_new, "da_gpu", &error);
	if (error != CL_SUCCESS) { std::cout << "Create kernel failed: " << error << std::endl; }
	group = 0;
	n = SIZE;
	x_d = clCreateBuffer(context_new, CL_MEM_READ_ONLY, sizeof(double) * SIZE, NULL, NULL);
	y_d = clCreateBuffer(context_new, CL_MEM_READ_WRITE, sizeof(double) * SIZE, NULL, NULL);
	error = clEnqueueWriteBuffer(queue_new, x_d, CL_TRUE, 0, sizeof(double) * SIZE, data_x_fcpu_d, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Enqueue write buffer data_x_fgpu_d failed: " << error << std::endl; }
	error = clEnqueueWriteBuffer(queue_new, y_d, CL_TRUE, 0, sizeof(double) * SIZE, data_y_fcpu_d, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Enqueue write buffer data_y_fgpu_d!!! failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 0, sizeof(int), &n);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for n failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 1, sizeof(double), &a_d);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for a failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 2, sizeof(cl_mem), &x_d);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for x failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 3, sizeof(int), &incx);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for incx failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 4, sizeof(cl_mem), &y_d);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for y failed: " << error << std::endl; }
	error = clSetKernelArg(kernel_new, 5, sizeof(int), &incy);
	if (error != CL_SUCCESS) { std::cout << "Set kernel args for incy failed: " << error << std::endl; }
	error = clGetKernelWorkGroupInfo(kernel_new, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
	if (error != CL_SUCCESS) { std::cout << "getKernelOwrkGroupInfo" << error << std::endl; }
	cl_event evt3;
	error = clEnqueueNDRangeKernel(queue_new, kernel_new, 1, NULL, &n, &localSize_gpu, 0, NULL, &evt3);
	if (error != CL_SUCCESS) { std::cout << "kernal work" << error << std::endl; }
	clWaitForEvents(1, &evt3);

	cl_ulong start_sd1 = 0, end_sd1 = 0;
	error = clGetEventProfilingInfo(evt3, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_sd1, nullptr);
	if (error != CL_SUCCESS) { std::cout << "Error getting start time here!: " << error << std::endl; }
	error = clGetEventProfilingInfo(evt3, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_sd1, nullptr);
	if (error != CL_SUCCESS) { std::cout << "Error getting end time: " << error << std::endl; }
	double* results_fcpu_d = new double[SIZE];
	error = clEnqueueReadBuffer(queue_new, y_d, CL_TRUE, 0, sizeof(double) * n, results_fcpu_d, 0, NULL, NULL);
	if (error != CL_SUCCESS) { std::cout << "Error readBuffer2: " << error << std::endl; }
	clFinish(queue_new);
	cpu_time_daxpy = end_sd1 - start_sd1;
	check_cpu_daxpy = checkResults(data_y_d, results_fcpu_d, SIZE);

	delete[] data_x_fcpu_d;
	delete[] data_y_fcpu_d;
	delete[] results_fcpu_d;
	clReleaseMemObject(x_d);
	clReleaseMemObject(y_d);
	clReleaseProgram(program_new);
	clReleaseKernel(kernel_new);
	clReleaseCommandQueue(queue_new);
	clReleaseContext(context_new);

	std::cout << "++++++++++++++++++DAXPY+++++++++++++++++++" << std::endl;
	std::cout << "+-----------------Check------------------+" << std::endl;
	std::cout << "+-------OMP--------Cpu---------GPU-------+" << std::endl;
	std::cout << "+--------" << check_omp_daxpy << "----------" << check_gpu_daxpy << "-----------" << check_cpu_daxpy << "--------+" << std::endl;
	std::cout << " Linear time " << linear_time_daxpy * (1e+03) << "ms" << std::endl;
	std::cout << " Omp time " << omp_time_daxpy * (1e+03) << "ms" << std::endl;
	std::cout << " GPU time " << (cl_double)(end_s1 - start_s1)*(cl_double)(1e-06) << "ms" << std::endl;
	std::cout << " CPU time " << (cl_double)(end_sd1 - start_sd1)*(cl_double)(1e-06) << "ms" << std::endl;
	//return 0;
}