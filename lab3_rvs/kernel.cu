#include <cuda.h>
#include "wb.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

int main(int argc, char* argv[]) {
	wbArg_t args;
	float* hostInput1;
	float* hostInput2;
	int inputLength;

	args = wbArg_read(argc, argv); // Чтение входных аргументов 
	wbTime_start(Generic, "Import data to host");
	hostInput1 = (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = (float*)wbImport(wbArg_getInputFile(args, 1), &inputLength);
	//wbTime_stop(Generic, "Importing data to host"); --> вылазит непонятная ошибка 


	// Объявление и выделение памяти под выходные данные
	float* hostOutput = (float*)malloc(sizeof(float) * inputLength);
	//@@ Место для вставки кода
	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	// Объявление и выделение памяти под входные и выходные данные  на устройства через thrust
	wbTime_start(GPU, "Doing GPU memory allocation");
	thrust::device_vector<float> input1(inputLength);
	thrust::device_vector<float> input2(inputLength);
	thrust::device_vector<float> output(inputLength);
	wbTime_start(Copy, "Copying data to the GPU");
	//@@ Место для вставки кода
	thrust::copy(hostInput1, hostInput1 + inputLength, input1.begin());
	thrust::copy(hostInput2, hostInput2 + inputLength, input2.begin());
	wbTime_stop(Copy, "Copying data to the GPU");

	// Выполнение операции сложения векторов
	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ Место для вставки кода
	thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::plus<float>());
	wbTime_stop(Compute, "Doing the computation on the GPU");
	/////////////////////////////////////////////////////////

	// Копирование данных обратно на хост
	wbTime_start(Copy, "Copying data from the GPU");
	//@@ Место для вставки кода
	thrust::copy(output.begin(), output.end(), hostOutput);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, hostOutput, inputLength);

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);
	return 0;
}