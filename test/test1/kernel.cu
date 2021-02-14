#include <CL/cl.h>
#include <iostream>

const char* source =
"__kernel void sum( __global float * input) \n"
"{										\n"
"	int i = get_global_id(0);			\n"
"	input[i] = input[i] + i;			\n"
"}";

const unsigned int SIZE = 1024;

int main() {
	cl_int error = 0;

	cl_uint num_platforms = 0;
	clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platform = NULL;
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

	cl_command_queue queue =
		clCreateCommandQueue(context, device, 0, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed" << std::endl;
	}

	size_t srclen[] = { strlen(source) };

	cl_program program =
		clCreateProgramWithSource(context, 1, &source, srclen, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create program with source failed" << std::endl;
	}

	clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	cl_kernel kernel = clCreateKernel(program, "sum", &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed" << std::endl;
	}

	size_t count = SIZE;
	float data[SIZE];
	float result[SIZE];
	size_t group = 0;

	for (int i = 0; i < SIZE; i++)
		data[i] = 10.0;

	cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * SIZE, NULL, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create buffer failed" << std::endl;
	}

	error = clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(float) * SIZE, data, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		std::cout << "Write buffer failed" << std::endl;
	}

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	if (error != CL_SUCCESS) {
		std::cout << "Set arg kernel failed" << std::endl;
	}

	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &((count)), &group, 0, NULL, NULL);

	clEnqueueReadBuffer(queue, input, CL_TRUE, 0, sizeof(float) * SIZE, result, 0, NULL, NULL);

	for (int i = 0; i < SIZE; i++)
	{
		printf("result[%d] = %f\n", i, result[i]);
	}

	clFinish(queue);
	clReleaseMemObject(input);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	return 0;
}

