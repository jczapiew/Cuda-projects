
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iomanip>

using namespace std;

void szeregLeibniza(float* tablica, int rozmiar)
{
	for (int i = 0; i < rozmiar; i++) {
		tablica[i] = pow(-1, i) * (1.0 / (2.0 * i + 1.0));
	}
}

float liczba_pi(float* tablica, int rozmiar) {
	float* tablica1 = new float[rozmiar];
	for (int i = 0; i < rozmiar; i++) {
		tablica1[i] = tablica[i];
	}
	for (int i = rozmiar / 2; i > 0; i = i / 2) {
		for (int j = 0; j < i; j++) {
			tablica1[j] = tablica1[2 * j] + tablica1[2 * j + 1];
		}
	}
	return tablica1[0];
}

void szeregLeibniza1(double* tablica, int rozmiar)
{
	for (int i = 0; i < rozmiar; i++) {
		tablica[i] = pow(-1, i) * (1.0 / (2.0 * i + 1.0));
	}
}

double liczba_pi1(double* tablica, int rozmiar) {
	double* tablica1 = new double[rozmiar];
	for (int i = 0; i < rozmiar; i++) {
		tablica1[i] = tablica[i];
	}
	for (int i = rozmiar / 2; i > 0; i = i / 2) {
		for (int j = 0; j < i; j++) {
			tablica1[j] = tablica1[2 * j] + tablica1[2 * j + 1];
		}
	}
	return tablica1[0];
}

__global__ void deviceLeibniz(float* tablica, size_t rozmiar)
{
	unsigned long long int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long long int siatka = blockDim.x * gridDim.x;
	for (i; i < rozmiar; i += siatka) {
		if (i % 2 == 0) tablica[i] = (1.0 / (2.0 * i + 1.0));
		else tablica[i] = (-1.0) * (1.0 / (2.0 * i + 1.0));
	}
}

__inline__ __device__ float redukcjaWarpow(float value) {
	for (int off = warpSize/2; off > 0; off /= 2)
		value += __shfl_down(value, off);
	return value;
}

__inline__ __device__ float redukcjaBlokow(float value) {

	static __shared__ float shared_mem[32]; // Shared mem for 32 partial sums
	int ll = threadIdx.x % warpSize;
	int ww = threadIdx.x / warpSize;

	value = redukcjaWarpow(value);     // Each warp performs partial reduction

	if (ll == 0) shared_mem[ww] = value; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	value = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[ll] : 0;

	if (ww == 0) value = redukcjaWarpow(value); //Final reduce within first warp

	return value;
}

__global__ void deviceAdd(float* tablica, float* wyjscie, size_t rozmiar)
{
	float sumka = 0;
	//reduce multiple elements per thread
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rozmiar; i += blockDim.x * gridDim.x)
	{
		sumka += tablica[i];
	}
	sumka = redukcjaBlokow(sumka);
	if (threadIdx.x == 0)
		wyjscie[blockIdx.x] = sumka;
}

__global__ void deviceLeibniz1(double* tablica, size_t rozmiar)
{
	unsigned long long int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long long int siatka = blockDim.x * gridDim.x;
	for (i; i < rozmiar; i += siatka) {
		if (i % 2 == 0) tablica[i] = (1.0 / (2.0 * i + 1.0));
		else tablica[i] = (-1.0) * (1.0 / (2.0 * i + 1.0));
	}
}

__inline__ __device__ double redukcjaWarpow1(double value) {
	for (int off = warpSize / 2; off > 0; off /= 2)
		value += __shfl_down(value, off);
	return value;
}

__inline__ __device__ double redukcjaBlokow1(double value) {

	static __shared__ double shared_mem[32]; // Shared mem for 32 partial sums
	int ll = threadIdx.x % warpSize;
	int ww = threadIdx.x / warpSize;

	value = redukcjaWarpow1(value);     // Each warp performs partial reduction

	if (ll == 0) shared_mem[ww] = value; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	value = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[ll] : 0;

	if (ww == 0) value = redukcjaWarpow1(value); //Final reduce within first warp

	return value;
}

__global__ void deviceAdd1(double* tablica, double* wyjscie, size_t rozmiar)
{
	double sumka = 0;
	//reduce multiple elements per thread
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rozmiar; i += blockDim.x * gridDim.x)
	{
		sumka += tablica[i];
	}
	sumka = redukcjaBlokow1(sumka);
	if (threadIdx.x == 0)
		wyjscie[blockIdx.x] = sumka;
}

int main()
{
	while (true) {
		int rozmiar;
		cout << "Podaj rozmiar wektora: ";
		cin >> rozmiar;
		cout << endl;
		float* tablica = new float[rozmiar];
		double* tablica1 = new double[rozmiar];

		clock_t start;
		double duration_on_CPU;
		double duration_on_CPU1;
		start = clock();

		szeregLeibniza(tablica, rozmiar);
		float wynik = 4.0 * liczba_pi(tablica, rozmiar);

		duration_on_CPU = 1000 * (clock() - start) / CLOCKS_PER_SEC;
		start = clock();

		szeregLeibniza1(tablica1, rozmiar);
		double wynik1 = 4.0 * liczba_pi1(tablica1, rozmiar);

		duration_on_CPU1 = 1000 * (clock() - start) / CLOCKS_PER_SEC;

		cout << "Obliczona liczba z szeregu Leibniza na CPU wynosi: " << setprecision(9) << wynik << " w prezycji float." << endl;
		cout << "A czas jej obliczenia wynosi: " << duration_on_CPU << " milisekund." << endl << endl;
		cout << "Obliczona liczba z szeregu Leibniza na CPU wynosi: " << setprecision(9) << wynik1 << " w precyzji double." << endl;
		cout << "A czas jej obliczenia wynosi: " << duration_on_CPU1 << " milisekund." << endl << endl;

		float host_wynik;
		double host_wynik1;
		float* dev_tablica;
		float* dev_wynik;
		double* dev_tablica1;
		double* dev_wynik1;

		int rozmiarBloku = 512;
		int liczbaBlokow = (rozmiar / rozmiarBloku) + 1;

		cudaMalloc((void**)&dev_tablica, rozmiar * sizeof(float));
		cudaMalloc((void**)&dev_wynik, liczbaBlokow * sizeof(float));
		cudaMalloc((void**)&dev_tablica1, rozmiar * sizeof(double));
		cudaMalloc((void**)&dev_wynik1, liczbaBlokow * sizeof(double));

		double duration_on_GPU;
		double duration_on_GPU1;
		start = clock();

		deviceLeibniz << <liczbaBlokow, rozmiarBloku >> > (dev_tablica, rozmiar);
		cudaDeviceSynchronize();
		deviceAdd << <liczbaBlokow, rozmiarBloku >> > (dev_tablica, dev_wynik, rozmiar);
		cudaDeviceSynchronize();
		deviceAdd << <1, 1024 >> > (dev_wynik, dev_tablica, liczbaBlokow);
		cudaDeviceSynchronize();

		duration_on_GPU = 1000 * (clock() - start) / CLOCKS_PER_SEC;

		cudaMemcpy(&host_wynik, dev_tablica, sizeof(float), cudaMemcpyDeviceToHost);

		start = clock();

		deviceLeibniz1 << <liczbaBlokow, rozmiarBloku >> > (dev_tablica1, rozmiar);
		cudaDeviceSynchronize();
		deviceAdd1 << <liczbaBlokow, rozmiarBloku >> > (dev_tablica1, dev_wynik1, rozmiar);
		cudaDeviceSynchronize();
		deviceAdd1 << <1, 1024 >> > (dev_wynik1, dev_tablica1, liczbaBlokow);
		cudaDeviceSynchronize();

		duration_on_GPU1 = 1000 * (clock() - start) / CLOCKS_PER_SEC;

		cudaMemcpy(&host_wynik1, dev_tablica1, sizeof(double), cudaMemcpyDeviceToHost);

		cout << "Obliczona liczba z szeregu Leibniza na GPU wynosi: " << setprecision(9) << 4.0 * host_wynik << " w prezycji float." << endl;
		cout << "A czas jej obliczenia wynosi: " << duration_on_GPU << " milisekund." << endl << endl;
		cout << "Obliczona liczba z szeregu Leibniza na GPU wynosi: " << setprecision(9) << 4.0 * host_wynik1 << " w precyzji double." << endl;
		cout << "A czas jej obliczenia wynosi: " << duration_on_GPU1 << " milisekund." << endl << endl;

		cout << endl << endl;

		cout << endl << endl;

		double stosunek;
		if (duration_on_GPU == 0) cout << "Czas obliczen w precyzji float na GPU jest zerowy i niemozliwe jest wyliczenie stosunku." << endl;
		else {
			stosunek = duration_on_CPU / duration_on_GPU;
			cout << "Obliczenia w precyzji float na GPU sa " << stosunek << " razy szybsze." << endl;
		}
		if (duration_on_GPU1 == 0) cout << "Czas obliczen w precyzji double na GPU jest zerowy i niemozliwe jest wyliczenie stosunku." << endl;
		else {
			stosunek = duration_on_CPU1 / duration_on_GPU1;
			cout << "Obliczenia w precyzji double na GPU sa " << stosunek << " razy szybsze." << endl;
		}
		cout << endl;

		cudaFree(dev_tablica);
		cudaFree(dev_wynik);
		cudaFree(dev_tablica1);
		cudaFree(dev_wynik1);
		delete[]tablica;
		delete[]tablica1;
	}
	return 0;
}