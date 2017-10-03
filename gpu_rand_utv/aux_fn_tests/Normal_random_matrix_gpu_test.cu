#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#define min( a,b ) ( (a) > (b) ? (b) : (a) )

__global__ 
static void Normal_random_number_kern( int m_A, int n_A, 
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

  Normal_random_number_kern<<< m_A * n_A, 1 >>>( m_A, n_A, A_pg, ldim_A, state, * rs_pt );

  * rs_pt += 1; 

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
  m_A = 5; n_A = 5; ldim_A = m_A;
  const char * A_name = "A";
  double * A_pc, * A_pg;


  curandState_t state; 
  unsigned long long rand_seed = 7;
  unsigned long long * rs_pt = & rand_seed;

  // allocate array on host (cpu)
  A_pc = ( double * ) malloc( m_A * n_A * sizeof( double ) );

  // allocate array on device (gpu)
  cudaMalloc( & A_pg, m_A * n_A * sizeof( double ) );

  // call function; fill gpu array with random standard normal numbers
  Normal_random_matrix( m_A, n_A, A_pg, ldim_A, state, rs_pt );
  
  // copy result to host
  cudaMemcpy( A_pc, A_pg, m_A * n_A * sizeof( double ), cudaMemcpyDeviceToHost );

  // print out result
  //print_double_matrix( A_name, m_A, n_A, A_pc, ldim_A );



  // repeat to test whether numbers generated are different

  // call function; fill gpu array with random standard normal numbers
  Normal_random_matrix( m_A, n_A, A_pg, ldim_A, state, rs_pt );

  // copy result to host
  cudaMemcpy( A_pc, A_pg, m_A * n_A * sizeof( double ), cudaMemcpyDeviceToHost );

  // print out result
  //print_double_matrix( A_name, m_A, n_A, A_pc, ldim_A );

  // free memory
  free( A_pc );
  cudaFree( A_pg );

  return 0;
}
