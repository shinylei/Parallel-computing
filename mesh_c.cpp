#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "f.h"

using namespace std;

int main(int argc, char** argv)
{
    int rank, size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double t1,t2;
	int n = 1000, m1, m2, i1_start, i1_end, i2_start, i2_end;
	int sqrtp = sqrt(size);
    int M = (int)ceil((double)n/sqrtp);
	long long A[M+1][M];
	long long buff1[M], buff2[M];
	long long sum, temp, medium;

	//STEP1: Initialize
	//m1, m2 is the length of squares storing data
    if (rank >= sqrtp * (sqrtp - 1)) {
	    m1 = n - (sqrtp - 1) * M;
	} else {
		m1 = M;
	}
	
	if (rank % sqrtp == sqrtp - 1) {
	    m2 = n - (sqrtp - 1) * M;
	} else {
	    m2 = M;
	}

	//determine the range need computation
    if (rank < sqrtp) {
		i1_start = 1;
	} else {
	    i1_start = 0;
	}

	if (rank >= sqrtp * (sqrtp - 1)) {
	    i1_end = m1 -1;
	} else {
	    i1_end = m1;
	}
	
    i2_start = 1;

    i2_end = m2-1;


	//initialize data matrix
	for (int i = 0; i < m1; i++){ 
	    for (int j = 0; j < m2; j++) {
		    A[i][j] = (rank / sqrtp * M + i) + (rank % sqrtp * M + j) * n;    
		}
	}
	if (rank % sqrtp > 0) {
	    for (int i = 0; i < m1; i++) {
		    buff1[i] = A[i][0]; 
		}
	}


	//STEP2: Call barrier and start timing
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0) t1 = MPI_Wtime();

	//STEP3: Do 10 iterations
	for (int step = 0; step < 10; step++) {
		if (rank > sqrtp - 1) {
		    MPI_Send(&A[0], m2, MPI_LONG_LONG, rank - sqrtp, 0, MPI_COMM_WORLD);
		}
		if (rank % sqrtp > 0) {
			MPI_Send(&buff1, m1, MPI_LONG_LONG, rank - 1, 1, MPI_COMM_WORLD);
		
		}
		if (rank > sqrtp -1 && rank % sqrtp > 0) {
			MPI_Send(&A[0][0], 1, MPI_LONG_LONG, rank - sqrtp - 1, 2, MPI_COMM_WORLD);
		}


		if (rank < size - sqrtp) {
		    MPI_Recv(&A[m1], m2, MPI_LONG_LONG, rank + sqrtp, 0, MPI_COMM_WORLD, &status);
		}
		if (rank % sqrtp < sqrtp - 1){
			MPI_Recv(&buff2, m1, MPI_LONG_LONG, rank + 1, 1, MPI_COMM_WORLD, &status);
		}
        if (rank < size - sqrtp && rank % sqrtp < sqrtp - 1) {
		    MPI_Recv(&buff2[m1], 1, MPI_LONG_LONG, rank + sqrtp + 1, 2, MPI_COMM_WORLD, &status);
		}


        if (rank % sqrtp > 0) {
		    for (int i = i1_start; i < i1_end-1; i++) {
				buff1[i] = f(buff1[i], buff1[i+1], A[i][1], A[i+1][1]);
			}
		    buff1[i1_end-1] = f(buff1[i1_end-1], A[i1_end][0], A[i1_end-1][1], A[i1_end][1]);
			A[0][0] = buff1[0];
		}
		for (int i = i1_start; i < i1_end; i++) {
		    for (int j = i2_start; j < i2_end; j++) {
		        A[i][j] = f(A[i][j], A[i+1][j], A[i][j+1], A[i+1][j+1]);	
			}
		}

		if (rank % sqrtp != sqrtp - 1) {
		    for (int i = i1_start; i < i1_end; i++) {
			    A[i][i2_end] = f(A[i][i2_end], A[i+1][i2_end], buff2[i], buff2[i+1]);
		    }
		}
	}



	//STEP4: Compute verification values
	//compute sum of all elements
	sum = 0;
	if (rank % sqrtp == 0) {
    	for (int i = 0; i < m1; i++) {
	        for (int j = 0; j < m2; j++) {
		        sum += A[i][j];
	    	}
    	}
	} else {
	    for (int i = 0; i < m1; i++) {
		    for (int j = 1; j < m2; j++) {
			    sum += A[i][j];
			}
		}

		for (int i = 0; i < m1; i++) {
		    sum += buff1[i];
		}
	}


    for (int i = 1; i < size; i = i * 2) {
	    if (rank % (i * 2) == i) {
		    MPI_Send(&sum, 1, MPI_LONG_LONG, rank - i, 3, MPI_COMM_WORLD);
		}
		if (rank % (i * 2) == 0 && rank + i < size) {
		    MPI_Recv(&temp, 1, MPI_LONG_LONG, rank + i, 3, MPI_COMM_WORLD, &status);
			sum = sum + temp;
		}
	}

	if (rank == n/2/M * sqrtp + n/2/M) {
		if (rank % sqrtp > 0 && n/2-n/2/M*M == 0) {
		    MPI_Send(&buff1[n/2-n/2/M*M], 1, MPI_LONG_LONG, 0, 4, MPI_COMM_WORLD);
		} else{
	        MPI_Send(&A[n/2-n/2/M*M][n/2-n/2/M*M], 1, MPI_LONG_LONG, 0, 4, MPI_COMM_WORLD);
	    }
	}

	if (rank == 0) {
	    MPI_Recv(&medium, 1, MPI_LONG_LONG, n/2/M*sqrtp+n/2/M, 4, MPI_COMM_WORLD, &status);
	}

	//STEP5: Program ends and print out time
	if(rank == 0){
    t2 = MPI_Wtime();
		cout<<"Result for mesh, 1000:"<<"\n";
        cout<<"Time:"<< t2 - t1 << "\n";
		cout<<"Sum of all elements:"<< sum <<"\n";
		cout<<"A[n/2][n/2]:"<< medium;
	}
	MPI_Finalize();
	return 0;
}
