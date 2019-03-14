#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "rand_utv_gpu.h"

#include <cuda_runtime.h>

#include <magma.h>
#include <magma_lapack.h>

#include <assert.h>
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
  double  * A_qr_h, * tau_h;
  double  * A_svd_h, * U_svd_h, * Vt_svd_h, * ss_h;
  double  * A_qrp_h;
  int     * jpvt_h;
  double  * B_h;
  double * A_mm_d, * B_d, * C_d;

  magma_int_t * magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );
  double      * work_h;
  int         lwork_qr, lwork_svd, lwork_qrp;
  int         * iwork_h;

  cudaError_t cudaStat = cudaSuccess;

  int i, j;

  int bl_size = 128;
  int n_A[] = {2000,3000,4000,5000,6000,8000,10000,12000,15000};
  int q = 2;
  int p = 0;

  // for timing
  timespec t1, t2;
  uint64_t diff;
  double   t_rutv_gpu[ sizeof( n_A ) / sizeof( int ) ],
		   t_dgeqrf_gpu[ sizeof( n_A ) / sizeof( int ) ], 
		   t_dgesdd_gpu[ sizeof( n_A ) / sizeof( int ) ],
		   t_dgeqp3_gpu[ sizeof( n_A ) / sizeof( int ) ],
		   t_dgemm_gpu[ sizeof( n_A ) / sizeof( int ) ];

  // for output file
  FILE * ofp;
  char mode = 'a';

  // initialize magma
  magma_init();
  
  for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {

	// allocate memory needed for this iteration 
	buff_A    = ( double * ) malloc( n_A[ i ] * n_A[ i ] *
									 sizeof( double ) );
	ldim_A    = max( 1, n_A[ i ] );
	buff_U    = ( double * ) malloc( n_A[ i ] * n_A[ i ] * 
									 sizeof( double ) );
	buff_V    = ( double * ) malloc( n_A[ i ] * n_A[ i ] * 
									sizeof( double ) );

	tau_h  = ( double * ) malloc( n_A[ i ] * sizeof( double ) );
	A_qr_h = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );
    
	A_svd_h  = ( double * ) malloc( n_A[ i ]*n_A[ i ] * sizeof( double ) );
	U_svd_h  = ( double * ) malloc( n_A[ i ]*n_A[ i ] * sizeof( double ) );
	Vt_svd_h = ( double * ) malloc( n_A[ i ]*n_A[ i ] * sizeof( double ) );
	ss_h     = ( double * ) malloc( n_A[ i ] * sizeof( double ) );

	A_qrp_h  = ( double * ) malloc( n_A[ i ]*n_A[ i ] * sizeof( double ) );
	jpvt_h   = ( int * ) malloc( n_A[ i ] * sizeof( int ) );

	B_h  = ( double * ) malloc( n_A[ i ]*n_A[ i ] * sizeof( double ) );
	cudaStat = cudaMalloc( & A_mm_d, n_A[ i ] * n_A[ i ] * sizeof( double ) );
	assert( cudaStat == cudaSuccess );
	cudaStat = cudaMalloc( & B_d, n_A[ i ] * n_A[ i ] * sizeof( double ) );
	assert( cudaStat == cudaSuccess );
	cudaStat = cudaMalloc( & C_d, n_A[ i ] * n_A[ i ] * sizeof( double ) );
	assert( cudaStat == cudaSuccess );
	
	// Generate and make copies of matrix A.
	matrix_generate( n_A[ i ], n_A[ i ], buff_A, ldim_A );
	matrix_generate( n_A[ i ], n_A[ i ], B_h, ldim_A );
	
	for ( j = 0; j < n_A[ i ]*n_A[ i ]; j++ ) {
	  A_qr_h[ j ] = buff_A[ j ]; 
	  A_svd_h[ j ] = buff_A[ j ];
	  A_qrp_h[ j ] = buff_A[ j ];
	}
	magma_setmatrix( n_A[ i ], n_A[ i ], sizeof( double ), 
					 buff_A, ldim_A, A_mm_d, ldim_A );
	magma_setmatrix( n_A[ i ], n_A[ i ], sizeof( double ),
					 B_h, ldim_A, B_d, ldim_A );
	
	if ( i == 0 ) {
	  // do one factorization before we start timing to get rid
	  // of "warmup" cost
	  rand_utv_gpu( n_A[i], n_A[i], buff_A, ldim_A,
					1, n_A[i], n_A[i], buff_U, n_A[i],
					1, n_A[i], n_A[i], buff_V, n_A[i],
					bl_size, p, q );
	}

	printf( "%% Working on n = %d \n", n_A[ i ] );

	// randUTV

		// start timing
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do factorization
		rand_utv_gpu( n_A[i], n_A[i], buff_A, ldim_A,
					  1, n_A[i], n_A[i], buff_U, n_A[i],
					  1, n_A[i], n_A[i], buff_V, n_A[i],
					  bl_size, p, q );
		
		// stop timing and record time
		cudaDeviceSynchronize();
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_rutv_gpu[ i ] = ( double ) diff / (1E9);

	// dgeqrf

	    // allocate work array
		lwork_qr = n_A[ i ] * magma_get_dgeqrf_nb( n_A[ i ], n_A[ i ] );
		work_h   = ( double * ) malloc( lwork_qr * sizeof( double ) );

		// start timing
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do factorization
		magma_dgeqrf( n_A[i], n_A[i], A_qr_h, ldim_A,
					  tau_h,
					  work_h, lwork_qr,
					  magInfo );
		if ( *magInfo != 0 ) {
		  printf("magma_dgeqrf failed! magInfo = %d \n", * magInfo );
		  return 1;
		}

		// generate Q matrix
		magma_dorgqr2( n_A[i], n_A[i], n_A[ i ], 
					   A_qr_h, ldim_A,
					   tau_h, 
					   magInfo );
		if ( *magInfo != 0 ) {
		  printf("magma_dorgqr2 failed! magInfo = %d \n", * magInfo );
		  return 1;
		}

		// stop timing and record time
		cudaDeviceSynchronize();
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_dgeqrf_gpu[ i ] = ( double ) diff / (1E9);

		// free work array
		free( work_h );

	// svd

	    // determine size of work array and allocate memory
		work_h   = ( double * ) malloc( sizeof( double ) );
		magma_dgesdd( MagmaAllVec, n_A[ i ], n_A[ i ],
					  NULL, ldim_A, NULL, NULL, ldim_A, NULL, ldim_A,
					  work_h, -1, NULL, magInfo ); // query to determine lwork_svd
		lwork_svd = ( int ) work_h[ 0 ];	
		free( work_h );
		work_h   = ( double * ) malloc( lwork_svd * sizeof( double ) );
		iwork_h  = ( int * ) malloc( 8 * n_A[ i ] );

		// start timing
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do factorization
		magma_dgesdd( MagmaAllVec, n_A[ i ], n_A[ i ],
					  A_svd_h, ldim_A, ss_h, U_svd_h, ldim_A, Vt_svd_h, ldim_A,
					  work_h, lwork_svd, iwork_h, magInfo );
		if ( *magInfo != 0 ) {
		  printf("magma_dgesdd failed! magInfo = %d \n", * magInfo );
		  return 1;
		}
		
		// stop timing and record time
		cudaDeviceSynchronize();
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_dgesdd_gpu[ i ] = ( double ) diff / (1E9);

		// free work arrays
		free( work_h );
		free( iwork_h );

	// dgeqp3

	    // allocate work array
		lwork_qrp = ( n_A[ i ] + 1 ) * magma_get_dgeqp3_nb( n_A[ i ], n_A[ i ] ) + 
					2 * n_A[ i ];
		work_h   = ( double * ) malloc( lwork_qrp * sizeof( double ) );

		// initialize pivot vector
		for ( j=0; j<n_A[ i ]; j++ ) {
		  jpvt_h[ j ] = 0;
		}

		// start timing
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, & t1 );

		// do factorization
		/*
		magma_dgeqp3( n_A[ i ], n_A[ i ], A_qrp_h, ldim_A,
					  jpvt_h, tau_h,
					  work_h, lwork_qrp, magInfo );
		if ( *magInfo != 0 ) {
		  printf("magma_dgeqp3 failed! magInfo = %d \n", * magInfo );
		  return 1;
		}
		*/
		// NOTE: need to do dgeqp3 in a separate file that can link with the ilp
		//       MKL library; currently, randutv is set up to link with the lp
		//       library only; right now, can get times for dgeqp3 from
		//       the script /repos/rnla_code/gpu_rand_utv/timing_standards.x

		// stop timing and record time
		cudaDeviceSynchronize();
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_dgeqp3_gpu[ i ] = ( double ) diff / (1E9);

		// free work array
		free( work_h );

	// dgemm

		// start timing
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do operation
		magma_dgemm( MagmaNoTrans, MagmaNoTrans, n_A[ i ], n_A[ i ], n_A[ i ],
					 1.0,
					 A_mm_d, ldim_A, B_d, ldim_A,
					 0.0,
					 C_d, ldim_A );

		// stop timing and record time
		cudaDeviceSynchronize();
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_dgemm_gpu[ i ] = ( double ) diff / (1E9);

	// Free matrices and vectors.
	free( buff_A );
	free( buff_U );
	free( buff_V );

	free( tau_h );
	free( A_qr_h );

	free( A_svd_h );
	free( U_svd_h );
	free( Vt_svd_h );
	free( ss_h );

	free( A_qrp_h );
	free( jpvt_h );

    free( B_h );
    cudaFree( A_mm_d );
	cudaFree( B_d );
	cudaFree( C_d );

  } // for i


  // write results to file
  ofp = fopen( "times_gpu.m", & mode );

	fprintf( ofp, "%% block size for randUTV was %d \n", bl_size );
	fprintf( ofp, "%% q value for randUTV was %d \n \n", q );
	fprintf( ofp, "%% p value for randUTV was %d \n \n", p );

	// write out vector of values of n used for these tests
	fprintf( ofp, "n_rutv_gpu = [ \n" );

	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp,  "%d ", n_A[ i ] );
	}

	fprintf( ofp, "]; \n \n");

    // write out vector of times 

    // randUTV
	fprintf( ofp, "t_rutv_gpu = [ \n" );

	for ( i=0; i < sizeof(n_A)/sizeof(int); i++ ) {
	  fprintf( ofp,  "%.2e ", t_rutv_gpu[ i ] );
	}
	fprintf( ofp, "]; \n \n");

    // dgeqrf
	fprintf( ofp, "t_dgeqrf_gpu = [ \n" );

	for ( i=0; i < sizeof(n_A)/sizeof(int); i++ ) {
	  fprintf( ofp,  "%.2e ", t_dgeqrf_gpu[ i ] );
	}
	fprintf( ofp, "]; \n \n");

    // dgesdd
	fprintf( ofp, "t_dgesdd_gpu = [ \n" );

	for ( i=0; i < sizeof(n_A)/sizeof(int); i++ ) {
	  fprintf( ofp,  "%.2e ", t_dgesdd_gpu[ i ] );
	}
	fprintf( ofp, "]; \n \n");

    // dgeqp3
	fprintf( ofp, "t_dgeqp3_gpu = [ \n" );

	for ( i=0; i < sizeof(n_A)/sizeof(int); i++ ) {
	  fprintf( ofp,  "%.2e ", t_dgeqp3_gpu[ i ] );
	}
	fprintf( ofp, "]; \n \n");

    // dgemm
	fprintf( ofp, "t_dgemm_gpu = [ \n" );

	for ( i=0; i < sizeof(n_A)/sizeof(int); i++ ) {
	  fprintf( ofp,  "%.2e ", t_dgemm_gpu[ i ] );
	}
	fprintf( ofp, "]; \n \n");

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

