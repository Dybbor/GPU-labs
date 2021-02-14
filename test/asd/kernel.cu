#include <CL/cl.h>
#include <iostream>

const char * source =
"__kernel void thread_printf() {"
" int group_id = get_group_id(0); \n"
" int local_id = get_local_id(0); \n"
" int global_id = get_global_id(0); \n"
" printf(\"I am from %d block, %d thread (global thread: %d)\\n\", \n"
" group_id, \n"
" local_id, \n"
" global_id); \n"
"}";

int main() {
	// Определение платформы
	cl_uint num_platforms = 0;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id platform = NULL;
	if (num_platforms > 0) {
		auto * platforms = new cl_platform_id[num_platforms];
		clGetPlatformIDs(num_platforms, platforms, NULL);
		platform = platforms[0];

		// Вывод информации о платформе
		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
		std::cout << platform_name << std::endl;

		delete[] platforms;
	}

	cl_int error;

	// Создание контекста
	cl_context_properties properties[3] = {
	CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
	};
	cl_context context = clCreateContextFromType(
		(platform == NULL) ? NULL : properties,
		CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create context failed" << std::endl;
	}

	// Определение устройства
	size_t num_devices = 0;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &num_devices);
	cl_device_id device = NULL;
	if (num_devices > 0) {
		auto * devices = new cl_device_id[num_devices];
		clGetContextInfo(context, CL_CONTEXT_DEVICES, num_devices, devices, NULL);
		device = devices[0];

		// Вывод информации о девайсе
		char device_name[128];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;

		delete[] devices;
	}

	// Создание очереди команд
	cl_command_queue queue = clCreateCommandQueue(
		context, device, 0, &error
	);
	if (error != CL_SUCCESS) {
		std::cout << "Create command queue failed" << std::endl;
	}

	// Создание программного объекта
	size_t src_len[] = { strlen(source) };
	cl_program program = clCreateProgramWithSource(
		context, 1, &source, src_len, &error
	);
	if (error != CL_SUCCESS) {
		std::cout << "Create program with source failed" << std::endl;
	}
	clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, "thread_printf", &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed" << std::endl;
	}

	// Определение размера
	size_t SIZE = 0;
	std::cout << "Enter size (multiple of 256)" << std::endl;
	std::cin >> SIZE;
	size_t count = SIZE;

	// Определение параметров запуска ядра
	size_t group;
	clGetKernelWorkGroupInfo(
		kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(size_t), &group, NULL);
	// Запуск ядра
	clEnqueueNDRangeKernel(
		queue, kernel, 1, NULL, &count, &group, 0, NULL, NULL);

	// Ожидание завершения команд в очереди
	clFinish(queue);

	// Освобождение ресурсов
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	system("pause");
	return 0;
}