#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "rand_utv_gpu.h"
#include <mkl.h>

#include <magma.h>

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
  double  * buff_A, * buff_Acp, * buff_U, * buff_Vt, * buff_ss, * buff_tau;
  magma_int_t * buff_p; // pivot vector for cpqr

  double * work_h;
  int lwork;
  magma_int_t * iwork_h;
  magma_int_t * magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );
  int nb;

  int i, j;

  int n_A[] = {1000,2000,3000,4000,5000,6000,8000,10000,12000};

  // for timing
  timespec t1, t2;
  uint64_t diff;
  double   t_cpqr[ sizeof( n_A ) / sizeof( int ) ];
  double   t_svd[ sizeof( n_A ) / sizeof( int ) ];

  // for output file
  FILE * ofp;
  char mode = 'w';

  // initialize magma
  magma_init();

  for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {

	// Create matrix A, matrix U, and matrix V.
	
	
	buff_A    = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );
	ldim_A    = max( 1, n_A[ i ] );

	buff_Acp    = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );

	buff_U    = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );

	buff_Vt   = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );

	buff_ss	  = ( double * ) malloc( n_A[ i ] * sizeof( double ) );

    buff_tau  = ( double * ) malloc( n_A[ i ] * sizeof( double ) );

    buff_p	  = ( magma_int_t * ) malloc( n_A[ i ] * sizeof( magma_int_t ) );

	// Generate matrix.
	matrix_generate( n_A[ i ], n_A[ i ], buff_A, ldim_A );

    // copy matrix since it's lost after dgeqp3 and dgesdd
	for ( j=0; j < n_A[i] * n_A[i]; j++ ) {
	  buff_Acp[ j ] = buff_A[ j ]; 
	}

	// Factorize matrix.
	printf( "%% Working on n = %d \n", n_A[ i ] );

	  // perform SVD with magma and record time
	  
		// determine size of work array
	    lwork = 3*n_A[i] + max( 3*n_A[i]*n_A[i] + 4*n_A[i], 2*n_A[i]*nb );
		nb = magma_get_dgesvd_nb( n_A[i], n_A[i] );

		// allocate memory
	    magma_dmalloc_pinned( & work_h, lwork );
		magma_imalloc_pinned( & iwork_h, 8 * n_A[i] );
  
		// start timing
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, & t1 );

		// do factorization
		magma_dgesdd( MagmaAllVec,
					  n_A[i], n_A[i], buff_A, ldim_A,
					  buff_ss,
					  buff_U, n_A[i],
					  buff_Vt, n_A[i],
					  work_h, lwork, iwork_h, magInfo );

		
		// stop timing and record time
		cudaDeviceSynchronize();
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_svd[ i ] = ( double ) diff / (1E9);

		// make sure factorization was successful
		if ( * magInfo != 0 ) {
		  printf("Warning! SVD failed \n");
		}


		// free work array so we can re-allocate it for cpqr
		magma_free_pinned( work_h );

	  // perform CPQR with magma
	  
	    // determine size of work array
	    nb = magma_get_dgeqp3_nb( n_A[i], n_A[i] );
		lwork = ( n_A[i] + 1 ) * nb + 2 * n_A[i];

		// allocate memory
	    magma_dmalloc_pinned( & work_h, lwork );

		// start timing
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, & t1 );

	    // initialize pivot vector to all zeros (default)
	    for ( j = 0; j < n_A[i]; j++ ) {
		  buff_p[ j ] = 0;
		}

		// do factorization
		magma_dgeqp3( n_A[i], n_A[i],
						buff_Acp, ldim_A,
						buff_p,
						buff_tau,
						work_h, lwork, magInfo );

		// stop timing and record time
		cudaDeviceSynchronize();
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_cpqr[ i ] = ( double ) diff / (1E9);

		// make sure factorization was successful
		if ( * magInfo != 0 ) {
		  printf( "Warning! CPQR failed \n" );
		}


	// Free matrices and vectors.
	free( buff_A );
	free( buff_Acp );
	free( buff_U );
	free( buff_Vt );
	free( buff_ss ); 
    free( buff_tau );

    free( buff_p );

	magma_free_pinned( work_h );
	magma_free_pinned( iwork_h );
  }

  // write results to file
  ofp = fopen( "times_std.m", & mode );

	// write out vector of values of n used for these tests
	fprintf( ofp, "n_std = [ \n" );

	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp,  "%d ", n_A[ i ] );
	}

	fprintf( ofp, "]; \n \n");


    // write out vector of times for svd

	fprintf( ofp, "t_svd = [ \n" );

	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp,  "%.2e ", t_svd[ i ] );
	}

	fprintf( ofp, "]; \n \n");

    // write out vector of times for cpqr
	fprintf( ofp, "t_cpqr = [ \n" );

	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp,  "%.2e ", t_cpqr[ i ] );
	}

	fprintf( ofp, "]; \n \n" );

  fclose(ofp);

  // finalize magma
  magma_finalize();

  free( magInfo );

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

