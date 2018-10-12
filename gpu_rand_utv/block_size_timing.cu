#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "rand_utv_gpu.h"
#include <mkl.h>

#include <time.h>

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )



// ============================================================================
// Declaration of local prototypes.

static void matrix_generate( int m_A, int n_A, 
				double * buff_A, int ldim_A );

// ============================================================================
int main() {
  
  int     ldim_A;
  double  * buff_A, * buff_U, * buff_V;

  int i, j;

  int bl_size_arr[] = {64, 128, 256, 512};
  int n_A = 18000; 
  int q[] = {0,2};
  int p = 0;

  // for timing
  timespec t1, t2;
  uint64_t diff;
  double   t_rutv_gpu[ (sizeof( n_A ) / sizeof( int ))*(sizeof(q)/sizeof(int)) ];

  // for output file
  FILE * ofp;
  char mode = 'a';
  
  printf( "%% n_A = %d \n", n_A );

  for ( j=0; j < sizeof( q ) / sizeof( int ); j++ ) {

	printf( "%% q = %d \n", q[j] );

	for ( i=0; i < sizeof( bl_size_arr ) / sizeof( int ); i++ ) {

	  // Create matrix A, matrix U, and matrix V.
	  buff_A    = ( double * ) malloc( n_A * n_A * sizeof( double ) );
	  ldim_A    = max( 1, n_A );

	  buff_U    = ( double * ) malloc( n_A * n_A * sizeof( double ) );

	  buff_V   = ( double * ) malloc( n_A * n_A * sizeof( double ) );

	  // Generate matrix.
	  matrix_generate( n_A, n_A, buff_A, ldim_A );

	  // Factorize matrix.
	  printf( "%% Working on bl_size = %d \n", bl_size_arr[ i ] );

		  // do factorization
		  rand_utv_gpu( n_A, n_A, buff_A, ldim_A,
						0, n_A, n_A, buff_U, n_A,
						0, n_A, n_A, buff_V, n_A,
						bl_size_arr[ i ], p, q[j] );
		  
	  // Free matrices and vectors.
	  free( buff_A );
	  free( buff_U );
	  free( buff_V );

	}

  }

  printf( "%% End of Program\n" );

  return 0;
}

// ============================================================================
static void matrix_generate( int m_A, int n_A, double * buff_A, int ldim_A ) {
  int     i, j;

  srand( 10 );
  for ( j = 0; j < n_A; j++ ) {
    for ( i = 0; i < m_A; i++ ) {
      buff_A[ i + j * ldim_A ] = ( double ) rand() / ( double ) RAND_MAX;
    }
  }
}

