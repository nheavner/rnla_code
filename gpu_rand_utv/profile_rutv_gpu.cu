#include <math.h>
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

  int bl_size = 512;
  int n_A[] = {2000,3000,4000,5000,6000,8000,10000,12000,15000};
  int q[] = {2};
  int p = 0;

  // for timing
  timespec t1, t2;
  uint64_t diff;
  double   t_rutv_gpu[ (sizeof( n_A ) / sizeof( int ))*(sizeof(q)/sizeof(int)) ];

  // for output file
  FILE * ofp;
  char mode = 'a';
  
  for ( j=0; j < sizeof( q ) / sizeof( int ); j++ ) {

	printf( "%% q = %d \n", q[j] );

	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {

	  // Create matrix A, matrix U, and matrix V.
	  buff_A    = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );
	  ldim_A    = max( 1, n_A[ i ] );

	  buff_U    = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );

	  buff_V   = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );

	  // Generate matrix.
	  matrix_generate( n_A[ i ], n_A[ i ], buff_A, ldim_A );

	  // Factorize matrix.
	  printf( "%% Working on n = %d \n", n_A[ i ] );

		  // do factorization
		  rand_utv_gpu( n_A[i], n_A[i], buff_A, ldim_A,
						1, n_A[i], n_A[i], buff_U, n_A[i],
						1, n_A[i], n_A[i], buff_V, n_A[i],
						bl_size, p, q[j] );
		  
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

