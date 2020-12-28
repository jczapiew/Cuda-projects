
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
using namespace std;

__global__ void primeKernel(unsigned long long int* liczb, bool* prawd)
{
	unsigned long long int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long long int siatka = blockDim.x * gridDim.x;

	for (i+=3; i * i <= liczb[0]; i += siatka) {
		if ((liczb[0] % i) == 0) {
			prawd[0] = false;
			break;
		}
	}
}

int main()
{
	while (true) {
		// 2305843009213693951 == 2^61 - 1
		unsigned long long int liczba[1];
		bool prawda[1];
		prawda[0] = true;

		cout << "Podaj liczbe calkowita: ";
		cin >> liczba[0];

		clock_t start;
		double duration_on_CPU;
		start = clock();
		if (liczba[0] % 2 != 0) {
			for (unsigned long long int i = 3; i * i <= liczba[0]; i += 2) {
				if (liczba[0] % i == 0) {
					prawda[0] = false;
					break;
				}
			}
		}
		else prawda[0] = false;
		if (liczba[0] == 2) prawda[0] = true;

		duration_on_CPU = 1000 * (clock() - start) / CLOCKS_PER_SEC;

		if (prawda[0]) cout << "Podana liczba jest liczba pierwsza na CPU." << endl;
		else cout << "Podana liczba nie jest liczba pierwsza na CPU." << endl;
		cout << "Obliczenia na CPU zajely " << duration_on_CPU << " mili sekund." << endl;

		prawda[0] = true;
		// Prime numbers in parallel
		unsigned long long int* dev_liczba;
		bool* dev_prawda;
		cudaMalloc((void**)& dev_liczba, sizeof(unsigned long long int));
		cudaMalloc((void**)& dev_prawda, sizeof(bool));
		cudaMemcpy(dev_liczba, liczba, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_prawda, prawda, sizeof(bool), cudaMemcpyHostToDevice);
		clock_t start2;
		double duration_on_GPU;
		start2 = clock();
		primeKernel << <256, 512 >> > (dev_liczba, dev_prawda);
		cudaDeviceSynchronize();
		duration_on_GPU = 1000 * (clock() - start2) / CLOCKS_PER_SEC;
		cudaMemcpy(prawda, dev_prawda, sizeof(bool), cudaMemcpyDeviceToHost);
		if (liczba[0] == 2) prawda[0] = true;

		if (prawda[0]) cout << "Podana liczba jest liczba pierwsza na GPU." << endl;
		else cout << "Podana liczba nie jest liczba pierwsza na GPU." << endl;
		cout << "Obliczenia na GPU zajely " << duration_on_GPU << " mili sekund." << endl;

		double stosunek;
		if (duration_on_GPU != 0) {
			stosunek = duration_on_CPU / duration_on_GPU;
			cout << "Obliczenia na GPU sa " << stosunek << " raza szybsze." << endl;
		}
		else {
			cout << "Obliczenie stosunku niemozliwe z powodu zerowego czasu GPU." << endl;
		}

		cudaFree(dev_liczba);
		cudaFree(dev_prawda);
	}
	return 0;
}
