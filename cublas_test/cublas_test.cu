#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include "dgemm_gpu.h"

// =========================================================================// Declaration of local prototypes

static void print_double_matrix ( const char * name, int m_A, int n_A,
				double * buff_A, int ldim_A );

// =========================================================================
int main() {

  // some initializations
  int i;
  double d_one = 1.0, d_zero = 0.0;
  char n = 'N', t = 'T';

  // Allocate 3 arrays on CPU
  int m_A, n_A, ldim_A, m_B, n_B, ldim_B, m_C, n_C, ldim_C;

  m_A = 4; n_A = 3; ldim_A = m_A; 
  m_B = 4; n_B = 5; ldim_B = m_B;
  m_C = n_A; n_C = n_B; ldim_C = n_A;
  const char * A_name = "A";
  const char * B_name = "B";
  const char * C_name = "C";

  double * buff_A = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  double * buff_B = ( double * ) malloc( m_B * n_B * sizeof( double ) );
  double * buff_C = ( double * ) malloc( m_C * n_C * sizeof( double ) );
  
  // Allocate 3 arrays on GPU
  double * buff_g_A, * buff_g_B, * buff_g_C;
  cudaMalloc( & buff_g_A, m_A * n_A * sizeof( double ) );
  cudaMalloc( & buff_g_B, m_B * n_B * sizeof( double ) );
  cudaMalloc( & buff_g_C, m_C * n_C * sizeof( double ) );

  // Initialize matrices A,B
  for ( i=0; i<m_A*n_A; i++ ) {
    buff_A[ i ] = ( double ) i;
  }

  for ( i=0; i<m_B*n_B; i++ ) {
    buff_B[ i ] = ( double ) i;
  }

  // print matrices A,B
  print_double_matrix( A_name, m_A, n_A, buff_A, ldim_A );
  print_double_matrix( B_name, m_B, n_B, buff_B, ldim_B );

  // transfer host arrays to device (gpu)
  cudaMemcpy( buff_g_A, buff_A, m_A * n_A * sizeof( double ), cudaMemcpyHostToDevice );
  cudaMemcpy( buff_g_B, buff_B, m_B * n_B * sizeof( double ), cudaMemcpyHostToDevice );

  // do the multiplication
  dgemm_gpu( t, n, n_A, n_B, m_B,
				& d_one, buff_g_A, ldim_A,
				buff_g_B, ldim_B,
				& d_zero, buff_g_C, ldim_C );

  // copy and print result on host memory
  cudaMemcpy( buff_C, buff_g_C, m_C * n_C * sizeof( double ), cudaMemcpyDeviceToHost );
  print_double_matrix( C_name, m_C, n_C, buff_C, ldim_C );

  // Free GPU memory
  cudaFree( buff_g_A );
  cudaFree( buff_g_B );
  cudaFree( buff_g_C );
  
  // Free CPU memory
  free( buff_A );
  free( buff_B );
  free( buff_C );

  return 0;

}

// =========================================================================
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
