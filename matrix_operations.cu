
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <time.h>

using namespace std;

class Matrix
{
private:
    int rozmiar_macierzy;
    double** macierz;
public:
    Matrix(int rozmiar)
    {
        rozmiar_macierzy = rozmiar;
        macierz = new double* [rozmiar];
        for (int i = 0; i < rozmiar; i++) {
            macierz[i] = new double[rozmiar];
        }
    }
	void free_memory();
	void random_values();
    int get_size();
    void set_value(int line, int column, double value);
    double get_value(int line, int column);
    Matrix transposition();
	void write_matrix();
};

void Matrix::free_memory()
{
	delete[] macierz;
}

void Matrix::random_values()
{
	int rozmiar = rozmiar_macierzy;
	for (int i = 0; i < rozmiar; i++) {
		for (int j = 0; j < rozmiar; j++) {
			macierz[i][j] = 2 * ((double)rand() / (double)RAND_MAX) - 1;
		}
	}
}

int Matrix::get_size()
{
    return rozmiar_macierzy;
}

void Matrix::set_value(int line, int column, double value)
{
    macierz[line][column] = value;
}

double Matrix::get_value(int line, int column)
{
    return macierz[line][column];
}

Matrix Matrix::transposition()
{
    int rozmiar = rozmiar_macierzy;
    Matrix At(rozmiar);

    for (int i = 0; i < rozmiar; i++) {
        for (int j = 0; j < rozmiar; j++) {
			At.set_value(i, j, macierz[j][i]);
        }
    }
    return At;
}

void Matrix::write_matrix()
{
	int rozmiar = rozmiar_macierzy;
	for (int i = 0; i < rozmiar; i++) {
		cout << "| ";
		for (int j = 0; j < rozmiar; j++) {
			cout << macierz[i][j] << " ";
		}
		cout << "|" << endl;
	}
	cout << endl;
}

Matrix multiplication(Matrix A, Matrix B)
{
    int rozmiar = A.get_size();
    double sum;
    Matrix C(rozmiar);
    
    for (int k = 0; k < rozmiar; k++) {
        for (int l = 0; l < rozmiar; l++) {
			sum = 0;
			for (int m = 0; m < rozmiar; m++) {
				sum = sum + A.get_value(k, m) * B.get_value(m, l);
            }
			C.set_value(k, l, sum);
        }
    }

    return C;
}

Matrix addition(Matrix A, Matrix B)
{
	int rozmiar = A.get_size();
	double sum;
	Matrix D(rozmiar);

	for (int i = 0; i < rozmiar; i++) {
		for (int j = 0; j < rozmiar; j++) {
			sum = A.get_value(i, j) + B.get_value(i, j);
			D.set_value(i, j, sum);
		}
	}

	return D;
}

void copy_values(Matrix macierz1, double macierz2[])
{
	int rozmiar = macierz1.get_size();
	for (int i = 0; i < rozmiar*rozmiar; i++) {
		macierz2[i] = macierz1.get_value(i/rozmiar, i%rozmiar);
	}
}

__global__ void deviceTransposition(double* macierz1, double* macierz2, int* rozmiar)
{
	unsigned long long int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long long int siatka = blockDim.x * gridDim.x;
	for (i; i < rozmiar[0]*rozmiar[0]; i += siatka) {
		macierz2[i] = macierz1[(i%rozmiar[0])*rozmiar[0]+i/rozmiar[0]];
	}
}

__global__ void deviceMultiplication(double* macierz1, double* macierz2, double* macierz3, int* rozmiar)
{
	unsigned long long int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long long int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned long long int siatkax = blockDim.x * gridDim.x;
	unsigned long long int siatkay = blockDim.y * gridDim.y;
	double sum;
	for (i; i < rozmiar[0] * rozmiar[0]; i += siatkax) {
		sum = 0;
		for (j; j < rozmiar[0]; j += siatkay) {
			sum = sum + macierz1[((i/rozmiar[0]) * rozmiar[0]) + j] * macierz2[i%rozmiar[0] + j*rozmiar[0]];
		}
		macierz3[i] = sum;
	}
}

__global__ void deviceAddition(double* macierz1, double* macierz2, double* macierz3, int* rozmiar)
{
	unsigned long long int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long long int siatka = blockDim.x * gridDim.x;
	for (i; i < rozmiar[0] * rozmiar[0]; i += siatka) {
		macierz3[i] = macierz2[i] + macierz1[i];
	}
}

int main()
{
	srand(time(NULL));
	while (true) {
		int rozmiar[1];
		cout << "Podaj rozmiar macierzy: ";
		cin >> rozmiar[0];

		Matrix A(rozmiar[0]);
		Matrix B(rozmiar[0]);
		Matrix C(rozmiar[0]);
		Matrix D(rozmiar[0]);

		A.random_values();
		B.random_values();

		/*cout << "macierz A:" << endl;
		A.write_matrix();
		cout << "macierz B:" << endl;
		B.write_matrix();*/

		clock_t start;
		double duration_on_CPU;
		start = clock();

		C = multiplication(A, B);
		/*cout << "Macierz C na CPU:" << endl;
		C.write_matrix();*/

		duration_on_CPU = 1000 * (clock() - start) / CLOCKS_PER_SEC;

		cout << "A*B na CPU zajelo: " << duration_on_CPU << " milisekund." << endl;

		double duration_on_CPU1;
		start = clock();

		D = addition(addition(multiplication(A, A.transposition()), multiplication(B, B.transposition())), multiplication(C, C.transposition()));
		/*cout << "Macierz D na CPU:" << endl;
		D.write_matrix();*/

		duration_on_CPU1 = 1000 * (clock() - start) / CLOCKS_PER_SEC;

		cout << "A*AT + B*BT + C*CT na CPU zajelo: " << duration_on_CPU1 << " milisekund." << endl;

		cout << "Ostatni element C: " << C.get_value(rozmiar[0] - 1, rozmiar[0] - 1) << endl;
		cout << "Ostatni element D: " << D.get_value(rozmiar[0] - 1, rozmiar[0] - 1) << endl;

		double* A1 = new double[rozmiar[0] * rozmiar[0]];
		double* B1 = new double[rozmiar[0] * rozmiar[0]];
		double* C1 = new double[rozmiar[0] * rozmiar[0]];
		double* D1 = new double[rozmiar[0] * rozmiar[0]];

		copy_values(A, A1);
		copy_values(B, B1);

		double* dev_A;
		double* dev_At;
		double* dev_B;
		double* dev_Bt;
		double* dev_C;
		double* dev_Ct;
		double* dev_D;
		double* dev_wynik;
		double* dev_wynik1;
		int* dev_rozmiar;
		
		int rozmiarBloku = 1024;
		int liczbaBlokow = (rozmiar[0] * rozmiar[0] + rozmiarBloku - 1) / rozmiarBloku;
		int sization = rozmiar[0] * rozmiar[0] * sizeof(double);

		cudaMalloc((void**)&dev_rozmiar, sizeof(int));
		cudaMalloc((void**)&dev_A, sization);
		cudaMalloc((void**)&dev_At, sization);
		cudaMalloc((void**)&dev_B, sization);
		cudaMalloc((void**)&dev_Bt, sization);
		cudaMalloc((void**)&dev_C, sization);
		cudaMalloc((void**)&dev_Ct, sization);
		cudaMalloc((void**)&dev_D, sization);
		cudaMalloc((void**)&dev_wynik, sization);
		cudaMalloc((void**)&dev_wynik1, sization);

		cudaMemcpy(dev_A, A1, sization, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_B, B1, sization, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_rozmiar, rozmiar, sizeof(int), cudaMemcpyHostToDevice);
		clock_t start2;
		double duration_on_GPU;
		start2 = clock();

		deviceMultiplication<<<liczbaBlokow, rozmiarBloku>>>(dev_A, dev_B, dev_C, dev_rozmiar);
		cudaDeviceSynchronize();

		duration_on_GPU = 1000 * (clock() - start2) / CLOCKS_PER_SEC;
		cudaMemcpy(C1, dev_C, sization, cudaMemcpyDeviceToHost);
		cout << "A*B na GPU zajelo: " << duration_on_GPU << " milisekund." << endl;

		/*cout << "Macierz C na GPU:" << endl;
		for (int i = 0; i < rozmiar[0] * rozmiar[0]; i++) {
			if (i % rozmiar[0] == 0) cout << endl;
			cout << C1[i] << " | ";
		}
		cout << endl;*/

		double duration_on_GPU1;
		start2 = clock();

		deviceTransposition<<<liczbaBlokow, rozmiarBloku>>>(dev_A, dev_At, dev_rozmiar);
		cudaDeviceSynchronize();
		deviceMultiplication<<<liczbaBlokow, rozmiarBloku>>>(dev_A, dev_At, dev_D, dev_rozmiar);
		cudaDeviceSynchronize();
		deviceTransposition<<<liczbaBlokow, rozmiarBloku>>>(dev_B, dev_Bt, dev_rozmiar);
		cudaDeviceSynchronize();
		deviceMultiplication<<<liczbaBlokow, rozmiarBloku>>>(dev_B, dev_Bt, dev_wynik, dev_rozmiar);
		cudaDeviceSynchronize();
		deviceTransposition<<<liczbaBlokow, rozmiarBloku>>>(dev_C, dev_Ct, dev_rozmiar);
		cudaDeviceSynchronize();
		deviceMultiplication<<<liczbaBlokow, rozmiarBloku>>>(dev_C, dev_Ct, dev_wynik1, dev_rozmiar);
		cudaDeviceSynchronize();
		deviceAddition<<<liczbaBlokow, rozmiarBloku>>>(dev_wynik, dev_D, dev_D, dev_rozmiar);
		cudaDeviceSynchronize();
		deviceAddition<<<liczbaBlokow, rozmiarBloku>>>(dev_wynik1, dev_D, dev_D, dev_rozmiar);
		cudaDeviceSynchronize();

		duration_on_GPU1 = 1000 * (clock() - start2) / CLOCKS_PER_SEC;
		cudaMemcpy(D1, dev_D, sization, cudaMemcpyDeviceToHost);
		cout << "A*AT + B*BT + C*CT na GPU zajelo: " << duration_on_GPU1 << " milisekund." << endl;

		cout << "Ostatni element C: " << C1[rozmiar[0] * rozmiar[0] - 1] << endl;
		cout << "Ostatni element D: " << D1[rozmiar[0] * rozmiar[0] - 1] << endl;

		/*cout << "Macierz D na GPU:" << endl;
		for (int i = 0; i < rozmiar[0] * rozmiar[0]; i++) {
			if (i % rozmiar[0] == 0) cout << endl;
			cout << D1[i] << " | ";
		}
		cout << endl;*/

		cout << endl;
		double stosunek;
		if (duration_on_GPU != 0) {
			stosunek = duration_on_CPU / duration_on_GPU;
			cout << "Pierwsze obliczenia na GPU sa " << stosunek << " raza szybsze." << endl;
		}
		else {
			cout << "Pierwszy czas na GPU jest zerowy, niemozliwe wyliczenie stosunku" << endl;
		}
		if (duration_on_GPU1 != 0) {
			stosunek = duration_on_CPU1 / duration_on_GPU1;
			cout << "Drugie obliczenia na GPU sa " << stosunek << " raza szybsze." << endl;
		}
		else {
			cout << "Drugi czas na GPU jest zerowy, niemozliwe wyliczenie stosunku" << endl;
		}
		cout << endl;

		double max = 0;
		for (int i = 0; i < rozmiar[0] * rozmiar[0]; i++) {
			if (abs(C.get_value(i / rozmiar[0], i % rozmiar[0]) - C1[i]) > max) max = abs(C.get_value(i / rozmiar[0], i % rozmiar[0]) - C1[i]);
		}
		if (max == 0) cout << "Nie ma roznicy miedzy macierzami C z CPU i GPU" << endl;
		else cout << "W miejscu najwiekszej rozbieznosci macierze C roznia sie o " << max << endl;

		max = 0;
		for (int i = 0; i < rozmiar[0] * rozmiar[0]; i++) {
			if (abs(D.get_value(i / rozmiar[0], i % rozmiar[0]) - D1[i]) > max) max = abs(D.get_value(i / rozmiar[0], i % rozmiar[0]) - D1[i]);
		}
		if (max == 0) cout << "Nie ma roznicy miedzy macierzami D z CPU i GPU" << endl;
		else cout << "W miejscu najwiekszej rozbieznosci macierze D roznia sie o " << max << endl;
		cout << endl;

		A.free_memory();
		B.free_memory();
		C.free_memory();
		D.free_memory();
		cudaFree(dev_rozmiar);
		cudaFree(dev_A);
		cudaFree(dev_B);
		cudaFree(dev_C);
		cudaFree(dev_D);
		cudaFree(dev_wynik);
		delete[] A1;
		delete[] B1;
		delete[] C1;
		delete[] D1;
	}
    return 0;
}