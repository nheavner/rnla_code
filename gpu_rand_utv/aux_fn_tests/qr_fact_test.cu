/*
    -L/usr/local/cuda/lib64 \
	-L/usr/local/magma/lib \
how to compile:
nvcc -c qr_fact_test.cu -I/opt/intel/mkl/incdlue -I/usr/local/magma/include -dc -DADD_ -DMAGMA_WITH_MKL
nvcc -o qr_fact_test.x qr_fact_test.o -L/usr/local/magma/lib -lmagma -lcusolver -lcublas -lgomp
*/

#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <magma.h>
#include <magma_lapack.h>

#define min( a,b ) ( (a) > (b) ? (b) : (a) )



__global__ 
static void Normal_random_matrix_kern( int m_A, int n_A, 
				double * A_pg, int ldim_A,
				curandState_t * states, 
				unsigned long long rand_seed ) {
  // fill matrix A with random numbers of standard normal distribution
 
  int ij = blockIdx.x * blockDim.x + threadIdx.x;
  
  int i = ij - m_A * ( ij / m_A ); // floor implicitly taken in div
  int j = ij / m_A; // floor implicitly taken
  
  //seed RNG
  curand_init( rand_seed, ij, 0, & states[ ij ] );

  if (i < m_A * n_A) {
    A_pg[ i + j * ldim_A ] = curand_normal( & states[ ij ]);
  }

}

static void Normal_random_matrix( int m_A, int n_A,
				double * A_pg, int ldim_A,
				curandState_t * states, 
				unsigned long long * rs_pt ) {
  // host function which fills the gpu array with random numbers
  // of standard normal distribution

  Normal_random_matrix_kern<<<m_A*n_A, 1 >>>( m_A, n_A, A_pg, ldim_A, states, * rs_pt );

  * rs_pt = * rs_pt + 1;

}

__global__
static void Make_upper_tri_kern( int m_A, int n_A,
				int ldim_A, double * A_pg ) {

  int ij = blockIdx.x * blockDim.x + threadIdx.x;
  
  // determine column, row indices for current thread
  int i = ij - m_A * ( ij / m_A ); // floor implicitly taken in div
  int j = ij / m_A; // floor implicitly taken

  // zero out elements below the main diagonal
  if ( ( ij < m_A * n_A ) && ( i > j ) ) {
    A_pg[ i + j * ldim_A ] = 0.0;
  }

}

static void Make_upper_tri( int m_A, int n_A,
				int ldim_A, double * A_pg ) {
  // given a matrix stored in device memory, 
  // this function sets all entries below the main diagonal to zero
  Make_upper_tri_kern<<< m_A*n_A, 1 >>>( m_A, n_A, ldim_A, A_pg );
}

static void gpu_dgeqrf( int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_pg ) {
  // given an m_A x n_A matrix A, calculates the QR factorization of A;
  // the HH vectors overwrite the lower tri portion of A
  
  // declare and initialize auxiliary variables
  double * work_pg = NULL; // work buffer array
  int lwork = 0; // size of work buffer
  cusolverDnHandle_t cusolverH = NULL; 
  int * devInfo = NULL; // stored in device

  // allocate space for devInfo on device
  cudaMalloc( ( void ** ) & devInfo, sizeof( int ) );

  // create cusolver handle
  cusolverDnCreate( & cusolverH );

  // determine size needed for workspace; this function just sets lwork 
  // to whatever it needs to be
  cusolverDnDgeqrf_bufferSize( cusolverH, 
				m_A, n_A, A_pg, ldim_A,
				& lwork );

  // now we can allocate memory for work array
  cudaMalloc( ( void ** ) & work_pg, sizeof( double ) * lwork );
  
  // compute factorization
  cusolverDnDgeqrf( cusolverH,
				m_A, n_A, A_pg, ldim_A,
				tau_pg, work_pg, lwork,
				devInfo );
  
  // free memory
  cudaFree( devInfo );
  cudaFree( work_pg );

  cusolverDnDestroy( cusolverH );

}


static void gpu_dormqr( char side, char op, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_pg,
				int m_C, int n_C, double * C_pg, int ldim_C ) {
  // calculates op(Q) * C or C * op(Q), where Q is the HH matrix from
  // the QR factorization of A; the HH vectors used to form Q must
  // be stored in the lower tri part of A_pg
  // 'side' tells you which side of the mult Q is on

  // declare and initialize auxiliary variables
  double * work_pg = NULL; // work buffer array
  int lwork = 0; // size of work buffer
  cusolverDnHandle_t cusolverH = NULL; 
  int * devInfo = NULL; // stored in device
  cublasSideMode_t cublas_side;
  cublasOperation_t cublas_op;


  // transfer side and op vars to proper cublas special types
  if ( side == 'L' ) {cublas_side = CUBLAS_SIDE_LEFT;}
  if ( side == 'R' ) {cublas_side = CUBLAS_SIDE_RIGHT;}

  if ( op == 'N' ) {cublas_op = CUBLAS_OP_N;}
  if ( op == 'T' ) {cublas_op = CUBLAS_OP_T;}
  

  // allocate space for devInfo on device
  cudaMalloc( ( void ** ) & devInfo, sizeof( int ) );

  // create cusolver handle
  cusolverDnCreate( & cusolverH );

  // determine size needed for workspace; this function just sets lwork 
  // to whatever it needs to be
  cusolverDnDormqr_bufferSize( cusolverH, cublas_side, cublas_op,
				m_C, n_C, min( m_A, n_A ), A_pg, ldim_A,
				tau_pg,
				C_pg, ldim_C,
				& lwork );

  // now we can allocate memory for work array
  cudaMalloc( ( void ** ) & work_pg, sizeof( double ) * lwork );
  
  // compute desired multiplication
  cusolverDnDormqr( cusolverH, cublas_side, cublas_op,
				m_C, n_C, min( m_A, n_A ), A_pg, ldim_A,
				tau_pg, 
				C_pg, ldim_C,
				work_pg, lwork, devInfo );
  
  // free memory
  cudaFree( devInfo );
  cudaFree( work_pg );

  cusolverDnDestroy( cusolverH );

}

void print_double_matrix( const char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A ) {
  int  i, j;

  printf( "%s = [\n", name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%le ", buff_A[ i + j * ldim_A ] );
    }
    printf( "\n" );
  }
  printf( "];\n" );
}

int main() {

  // declare, initialize variables
  int m_A, n_A, ldim_A;
  m_A = 10000; n_A = 128; ldim_A = m_A;
  const char * A_name = "A";
  double * A_pc, * A_pg, * A_pgc;
  char n = 'N', t = 'T', l = 'L', r = 'R';
  double * tau_h = NULL; // tau is the scaling factor for each HH vector
						  // such that H = I - tau*q*q', where q is the 
						  // vector stored in the lower portion of A after
						  // factorization
  magma_int_t * magInfo = (magma_int_t * ) malloc( sizeof( magma_int_t ) );

  curandState_t * states; // we store a random state for every thread
  unsigned long long rand_seed = 7;
  unsigned long long * rs_pt = & rand_seed;

  magma_init();

  // allocate space on GPU for the random states
  cudaMalloc( (void **) & states, m_A * n_A * sizeof( curandState_t ) );

  // allocate array on host (cpu)
  A_pc = ( double * ) malloc( m_A * n_A * sizeof( double ) );

  // allocate array on device (gpu)
  cudaMalloc( & A_pg, m_A * n_A * sizeof( double ) );
  cudaMalloc( & A_pgc, m_A * n_A * sizeof( double ) );

  // allocate tau array
  tau_h = (double *) malloc( m_A * sizeof( double ) );

  // fill gpu array with random standard normal numbers
  Normal_random_matrix( m_A, n_A, A_pg, ldim_A, states, rs_pt );
  
  // check
  cudaMemcpy( A_pc, A_pg, m_A * n_A * sizeof( double ), cudaMemcpyDeviceToHost );
  //print_double_matrix( A_name, m_A, n_A, A_pc, ldim_A );
  
  // compute QR factorization
  magma_dgeqrf2_gpu( m_A, n_A, A_pg, ldim_A, tau_h, magInfo );
	
  // copy result so we can check error
  cudaMemcpy( A_pgc, A_pg, m_A * n_A * sizeof( double ), cudaMemcpyDeviceToDevice );

  // make copy upper tri so it's the R factor in QR
  Make_upper_tri( m_A, n_A, ldim_A, A_pgc );

  // print out R factor to make sure it's upper tri
  //cudaMemcpy( A_pc, A_pgc, m_A * n_A * sizeof( double ), cudaMemcpyDeviceToHost );
  //print_double_matrix( A_name, m_A, n_A, A_pc, ldim_A );

  // do multiplication to get original matrix 
  //gpu_dormqr( l, n, 
	//			m_A, n_A, A_pg, ldim_A,
	//			tau_pg,
	//			A_pgc, ldim_A );
  
  // copy result to host
  //cudaMemcpy( A_pc, A_pgc, m_A * n_A * sizeof( double ), cudaMemcpyDeviceToHost );

  // print out result
  //print_double_matrix( A_name, m_A, n_A, A_pc, ldim_A );

  magma_finalize();

  // free memory
  free( A_pc );
  cudaFree( A_pg );
  cudaFree( A_pgc );
  cudaFree( states );

  free( magInfo );
  free( tau_h );

  return 0;
}
