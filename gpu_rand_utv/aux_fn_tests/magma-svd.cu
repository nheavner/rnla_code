/*
how to compile:
nvcc -I/usr/local/magma/include -c magma-svd.cu
nvcc -o magma-svd.x magma-svd.o -L/usr/local/magma/lib \
	-lmagma -lcusolver -lcublas -lgomp
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <magma.h>

#define min( a,b ) ( (a) > (b) ? (b) : (a) )
#define max( a,b ) ( (a) > (b) ? (a) : (b) )

static void gpu_print_double_matrix( char * name, int m_A, int n_A,  
               double * buff_A, int ldim_A ) { 
 int  i, j; 

 double * A_pc; 
 
 A_pc = ( double * ) malloc( m_A * n_A * sizeof( double ) ); 
 
 cudaMemcpy( A_pc, buff_A, m_A * n_A * sizeof( double ), 
          cudaMemcpyDeviceToHost ); 

   printf( "%s = [\n", name ); 
   for( i = 0; i < m_A; i++ ) { 
     for( j = 0; j < n_A; j++ ) { 
     printf( "%.16e ", A_pc[ i + j * ldim_A ] ); 
   } 
   printf( "\n" ); 
  } 
  printf( "];\n" ); 
  
  free( A_pc ); 
} 

static void print_double_matrix(const char * name, int m_A, int n_A, 
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

__global__ 
static void Normal_random_matrix_kern( int m_A, int n_A, 
				double * A_pg, int ldim_A,
				curandState_t state, 
				unsigned long long rand_seed ) {
  // fill matrix A with random numbers of standard normal distribution
 
  int ij = blockIdx.x * blockDim.x + threadIdx.x;
  
  int i = ij - m_A * ( ij / m_A ); // floor implicitly taken in div
  int j = ij / m_A; // floor implicitly taken
  
  //seed RNG
  curand_init( rand_seed, ij, 0, & state );

  if ( ij < m_A * n_A ) {
    A_pg[ i + j * ldim_A ] = curand_normal( & state );
  }

}

static void Normal_random_matrix( int m_A, int n_A,
				double * A_pg, int ldim_A,
				curandState_t state, 
				unsigned long long * rs_pt ) {
  // host function which fills the gpu array with random numbers
  // of standard normal distribution

  Normal_random_matrix_kern<<<m_A*n_A, 1 >>>( m_A, n_A, A_pg, ldim_A, state, * rs_pt );

  * rs_pt = * rs_pt + 1;

}

__global__
static void Set_ss_diag_mat_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * ss_pg ) {
  // kernel function which sets matrix represented by A_pg to a diagonal
  // matrix with the svs of A on the diagonal

  int ij = blockIdx.x * blockDim.x + threadIdx.x;
  
  // determine column, row indices for current thread
  int i = ij - m_A * ( ij / m_A ); // floor implicitly taken in div
  int j = ij / m_A; // floor implicitly taken

  // fill in matrix
  if ( ( ij < m_A * n_A ) && ( i == j ) ) {
    A_pg[ i + j * ldim_A ] = ss_pg[ i ];
  }
  else if ( ij < m_A * n_A ) {
    A_pg[ i + j * ldim_A ] = 0.0; 
  }
  
}

static void Set_ss_diag_mat( int m_A, int n_A, double * A_pg, int ldim_A,
				double * ss_pg ) {
  // host function which sets matrix represented by A_pg to a diagonal
  // matrix with the svs of A on the diagonal
  Set_ss_diag_mat_kern<<< ( m_A * n_A / 256 ) + 1, 256 >>>( m_A, n_A, A_pg, ldim_A,
				ss_pg );
}

static void local_magsvd( int m_A, int n_A, double * A_pg, int ldim_A,
				double * ss_pg,
				double * U_pg, int ldim_U,
				double * Vt_pg, int ldim_Vt ) {
  // given an m_A x n_A matrix A stored in device memory in A_pg,
  // this function computes the svd on the device
  
  // declare and initialize auxiliary variables
  
  double * ss_p;
  double * A_p, * U_p, * Vt_p;
  
  double * work_p = NULL; // work buffer array
  int lwork = 0; // size of work buffer
  int * iwork_p; // an integer workspace array
  magma_int_t * magInfo = NULL; // stored in device
  
  // vars for determining size of work array
  int nb = magma_get_dgesvd_nb( m_A, n_A );
  int A_mx, A_mn;

  // get max and min
  A_mx = max( m_A, n_A );
  A_mn = min( m_A, n_A );

  // allocate space for devInfo on device
  magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );

  // determine size of work array
  // all SVDs for randUTV will be square, which determines our
  // equation for finding lwork
  lwork = 3*A_mn + max( 3*A_mn*A_mn + 4*A_mn, (A_mx+A_mn)*nb );
  
  // now we can allocate memory for work arrays
  magma_dmalloc_pinned( & work_p, lwork );
  magma_imalloc_pinned( & iwork_p, 8 * A_mn );

  // arrays must be in host memory for magma svd
  A_p = ( double * ) malloc( m_A * n_A * sizeof( double ) ); 
  U_p = ( double * ) malloc( m_A * m_A * sizeof( double ) ); 
  Vt_p = ( double * ) malloc( n_A * n_A * sizeof( double ) ); 
  ss_p = ( double * ) malloc( A_mn * sizeof( double ) ); 
  
  cudaMemcpy( A_p, A_pg, m_A * n_A * sizeof( double ),
				cudaMemcpyDeviceToHost );
  cudaMemcpy( U_p, U_pg, m_A * m_A * sizeof( double ),
				cudaMemcpyDeviceToHost );
  cudaMemcpy( Vt_p, Vt_pg, n_A * n_A * sizeof( double ),
				cudaMemcpyDeviceToHost );

  // compute factorization
  magma_dgesdd( MagmaAllVec,
				m_A, n_A, A_p, ldim_A,
				ss_p, 
				U_p, ldim_U,
				Vt_p, ldim_Vt,
				work_p, lwork, iwork_p, magInfo );
  
  // transfer results back to device
  cudaMemcpy( U_pg, U_p, m_A * m_A * sizeof( double ),
				cudaMemcpyHostToDevice );
  cudaMemcpy( Vt_pg, Vt_p, n_A * n_A * sizeof( double ),
				cudaMemcpyHostToDevice );
  cudaMemcpy( ss_pg, ss_p, A_mn * sizeof( double ),
				cudaMemcpyHostToDevice );

  // set contents of A_pg to zeros with svs on diagonal
  Set_ss_diag_mat( m_A, n_A, A_pg, ldim_A, ss_pg );

  // free memory
  free( A_p );
  free( U_p );
  free( Vt_p );
  free( ss_p );
   
  free( magInfo );
  magma_free_pinned( work_p );
  magma_free_pinned( iwork_p );
  
}

static void gpu_dgemm( char opA, char opB, double alpha, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				int m_B, int n_B, double * B_pg, int ldim_B,
				double beta,
				int m_C, int n_C, double * C_pg, int ldim_C ) {

  // generate the correct transpose option identifier that CUBLAS accepts
  // also determine the correct "middle" dim of the mult
  cublasOperation_t cutransA, cutransB;
  int middle_dim;

  if ( opA == 'N' ) { cutransA = CUBLAS_OP_N; middle_dim = n_A; }
  else if ( opA == 'T' ) { cutransA = CUBLAS_OP_T; middle_dim = m_A; }

  if ( opB == 'N' ) { cutransB = CUBLAS_OP_N; }
  else if ( opB == 'T' ) { cutransB = CUBLAS_OP_T; }


  // create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate( & handle );

  // do the multiplication
  cublasDgemm( handle, cutransA, cutransB,
				m_C, n_C, middle_dim, & alpha,
				A_pg, ldim_A, B_pg, ldim_B, 
				& beta, C_pg, ldim_C );

  // destroy the handle
  cublasDestroy( handle );

}




int main() {

  // declare, initialize variables
  int m_A, n_A, ldim_A;
  m_A = 300; n_A = 300; ldim_A = m_A;
  const char * A_name = "A";
  double * A_pc, * A_pg, * A_pgc;
  
  int i;
  double err = 0.0, A_norm = 0.0; // for checking factorization later

  char n = 'N', t = 'T';
  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0;
  double * ss_pg;
  int m_U, n_U, ldim_U;
  m_U = m_A; n_U = m_U; ldim_U = m_U;
  double * U_pg;
  int m_Vt, n_Vt, ldim_Vt;
  m_Vt = n_A; n_Vt = m_Vt; ldim_Vt = m_Vt;
  double * Vt_pg;

  curandState_t state; // we store a random state for every thread
  unsigned long long rand_seed = 7;
  unsigned long long * rs_pt = & rand_seed;

  cudaError_t cudaStat = cudaSuccess;
 
  // initialize magma
  magma_init();

  // allocate array on host (cpu)
  A_pc = ( double * ) malloc( m_A * n_A * sizeof( double ) );

  // allocate array on device (gpu)
  cudaStat = cudaMalloc( ( void ** ) & A_pg, m_A * n_A * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
 

  cudaStat = cudaMalloc( ( void ** ) & A_pgc, m_A * n_A * sizeof( double ) );
  assert( cudaStat == cudaSuccess );


  // allocate arrays for svd output
  cudaStat = cudaMalloc( ( void ** ) & ss_pg, n_A * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  
  cudaStat = cudaMalloc( ( void ** ) & U_pg, m_U * n_U * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  
  cudaStat = cudaMalloc( ( void ** ) & Vt_pg, m_Vt * n_Vt * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  // fill gpu array with random standard normal numbers
  Normal_random_matrix( m_A, n_A, A_pg, ldim_A, state, rs_pt );
 
  // copy matrix so we can check error later
  cudaMemcpy( A_pgc, A_pg, m_A * n_A * sizeof( double ), cudaMemcpyDeviceToDevice );

  // check
  cudaMemcpy( A_pc, A_pg, m_A * n_A * sizeof( double ), cudaMemcpyDeviceToHost );
  //print_double_matrix( A_name, m_A, n_A, A_pc, ldim_A );
  
  // compute norm to check relative error later
  for ( i=0; i < m_A * n_A; i++ ) {
    A_norm += pow( A_pc[ i ], 2 );
  }
  A_norm = sqrt( A_norm );

  // compute SVD factorization
  local_magsvd( m_A, n_A, A_pg, ldim_A,
				ss_pg,
				U_pg, ldim_U,
				Vt_pg, ldim_Vt );
  
  // compute U*D*Vt to check factorization
  gpu_dgemm( n, n, d_one, 
				m_U, n_U, U_pg, ldim_U,
				m_A, n_A, A_pg, ldim_A,
				d_zero,
				m_A, n_A, A_pg, ldim_A );
  
  gpu_dgemm( n, n, d_one, 
				m_A, n_A, A_pg, ldim_A,
				m_Vt, n_Vt, Vt_pg, ldim_Vt,
				d_neg_one,
				m_A, n_A, A_pgc, ldim_A );


  // copy result to host
  cudaMemcpy( A_pc, A_pgc, m_A * n_A * sizeof( double ), cudaMemcpyDeviceToHost );

  // compute relative error
  for ( i=0; i < m_A * n_A; i++ ) {
    err += pow( A_pc[ i ], 2 );
  }
  err = sqrt( err );
  

  // print out result
  //print_double_matrix( A_name, m_A, n_A, A_pc, ldim_A );
  printf( "%% ||A - UDV'||_F / ||A||_F = %e \n", err );

  // finalize magma
  magma_finalize();

  // free memory
  free( A_pc );
  cudaFree( A_pg );

  cudaFree( ss_pg );
  cudaFree( U_pg );
  cudaFree( Vt_pg );

  return 0;
}
