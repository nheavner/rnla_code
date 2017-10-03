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

  int bl_size = 128;
  int n_A[] = {1000,2000,3000,4000,5000,6000,8000,10000,12000};
  int q[] = {0,1,2};

  // for timing
  timespec t1, t2;
  uint64_t diff;
  double   t_rutv_gpu[ (sizeof( n_A ) / sizeof( int ))*(sizeof(q)/sizeof(int)) ];

  // for output file
  FILE * ofp;
  char mode = 'w';
  
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

		  // start timing
		  cudaDeviceSynchronize();
		  clock_gettime(CLOCK_MONOTONIC, & t1 );
		  
		  // do factorization
		  rand_utv_gpu( n_A[i], n_A[i], buff_A, ldim_A,
						1, n_A[i], n_A[i], buff_U, n_A[i],
						1, n_A[i], n_A[i], buff_V, n_A[i],
						bl_size, 0, q[j] );
		  
		  // stop timing and record time
		  cudaDeviceSynchronize();
		  clock_gettime( CLOCK_MONOTONIC, & t2 );
		  diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		  t_rutv_gpu[ i + j*(sizeof(n_A)/sizeof(int)) ] = ( double ) diff / (1E9);

	  // Free matrices and vectors.
	  free( buff_A );
	  free( buff_U );
	  free( buff_V );

	}

  }

  // write results to file
  ofp = fopen( "times_rutv_gpu.m", & mode );

	fprintf( ofp, "%% block size was %d \n \n", bl_size );

	fprintf( ofp, "%% the ROWS of the matrix t_rutv_gpu correspond to the values of q in ascending order (i.e. 1st row is q=0) \n \n " );

	// write out vector of values of n used for these tests
	fprintf( ofp, "n_rutv_gpu = [ \n" );

	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp,  "%d ", n_A[ i ] );
	}

	fprintf( ofp, "]; \n \n");

    // write out vector of times 

	fprintf( ofp, "t_rutv_gpu = [ \n" );

	for ( i=0; i < (sizeof(n_A) * sizeof(q)) / (sizeof(int) * sizeof(int)); i++ ) {
	  fprintf( ofp,  "%.2e ", t_rutv_gpu[ i ] );

	  if ( (i+1) % (sizeof(n_A)/sizeof(int)) == 0 ) 
	    fprintf( ofp, "; \n" );
	}

	fprintf( ofp, "]; \n \n");

  fclose(ofp);

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

