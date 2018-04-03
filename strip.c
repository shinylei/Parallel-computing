#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "f.h"



void main(int argc, char** argv)
{
    int rank, size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	double t1,t2;
	int n = 1000, m, i_start, i_end;         
	int M = (int)ceil((double)n/size);
	long long A[M+1][n];
	long long sum, temp, medium;

    //STEP1: Initialize
	//m is #rows for each proc
	if (rank == size-1) {
	    m = n-M*(size-1);
	} else {
	    m = M;
	}

	//define the range need computation
	if (rank == 0) {
		i_start = 1;
	} else {
	    i_start = 0;
	}

	if (rank == size -1) {
	    i_end = m - 1;
	} else {
	    i_end = m;
	}

	//initalize data matrix
	for (int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			A[i][j] =  (rank*M + i) + j*n;
		}
	}



	//STEP2: Call barrier and start timing
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0) t1 = MPI_Wtime();


    //STEP3: Do 10 iterations
	for (int step = 0; step < 10; step++) {
	if (rank > 0) {
		    MPI_Send(&A[0], n, MPI_LONG_LONG, rank-1, 0, MPI_COMM_WORLD);
		}
		if (rank < size -1) {
		    MPI_Recv(&A[m], n, MPI_LONG_LONG, rank+1, 0, MPI_COMM_WORLD, &status);
		}
        
    	for (int i = i_start; i < i_end; i++) {
	    	for (int j = 1; j < n-1; j++) {
                A[i][j] = f(A[i][j], A[i+1][j], A[i][j+1], A[i+1][j+1]); 
	    	}
    	}
	}    


	//STEP4: Compute verification values
	//compute sum of all elements
	sum = 0;
	for (int i = 0; i < m; i++) {
	    for (int j = 0; j < n; j++) {
		    sum += A[i][j];
		}
	}
	//printf("rank:%d, sum: %lld\n",rank,sum);

    
	for (int i = 1; i < size; i = i*2) {
	    if(rank % (i*2) == i) {
		    MPI_Send(&sum, 1, MPI_LONG_LONG, rank-i, 1, MPI_COMM_WORLD);
			//printf("send from %d to %d: %lld\n",rank, rank-i, sum);
		}
		if(rank % (i*2) == 0 && rank+i < size) {
		    MPI_Recv(&temp, 1, MPI_LONG_LONG, rank+i, 1, MPI_COMM_WORLD, &status);
			sum = sum + temp;
			//printf("%d recieve data from %d: data_receive:%lld, sum:%lld\n",rank,rank+i,temp,sum);
		}
	}

	if (rank == (n/2 + 1)/M) {
	    MPI_Send(&A[n/2-(n/2+1)/M*M][n/2], 1, MPI_LONG_LONG, 0, 2, MPI_COMM_WORLD);
	}

	if (rank == 0) {
	    MPI_Recv(&medium, 1, MPI_LONG_LONG, (n/2 + 1)/M, 2, MPI_COMM_WORLD, &status);
	}
	
    //STEP5: Program ends and print out time
    if(rank == 0)
	{
		t2 = MPI_Wtime();
		printf("Time: %f\n",t2-t1);
		printf("Sum of all elements: %lld\n",sum);
		printf("A[n/2][n/2]: %lld\n", medium);
	
	}
    MPI_Finalize();
}

