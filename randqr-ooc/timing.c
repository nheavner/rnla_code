#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <mkl.h>

#include <time.h>
#include "hqrrp_ooc.h"
#include "NoFLA_HQRRP_WY_blk_var4.h"

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )



// ============================================================================
// Declaration of local prototypes.

static void matrix_generate_ooc( int m_A, int n_A, char * A_fname );

// ============================================================================
int main() {
  
  int     ldim_A;
  double  * buff_A, * buff_tau;
  int  * buff_p;
  char A_ssd_fname[] = "./A_mat_ssd";
  char A_hdd_fname[] = "/media/hdd/A_mat_hdd";

  size_t read_check;

  int i, j;

  int bl_size = 250;
  int k = 1000;
  int p = 0;
  int n_A[] = {150000};//{1000, 2000, 4000, 5000, 8000, 10000, 15000, 20000, 30000, 40000, 45000, 50000, 70000, 100000, 110000, 120000, 150000};


  // for timing
  struct timespec t1, t2;
  uint64_t diff;
  double   t_cpqr_ssd[ (sizeof( n_A ) / sizeof( int )) ];
  double   t_cpqr_hdd[ (sizeof( n_A ) / sizeof( int )) ];
  double   t_cpqr_in[ (sizeof( n_A ) / sizeof( int )) ];

  // for output file
  FILE * ofp, * A_fp;
  char mode = 'a';

  for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {

	// Create matrix A, matrix P, and vector tau.
	if ( n_A[ i ] <= 45000 ) {
	  buff_A    = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );
      if ( !buff_A ) {
	    printf("Error! Memory allocation failed \n"); 
	  }
	}

	buff_p    = ( int * )  malloc( n_A[ i ] * sizeof( int ) ); 

	buff_tau  = ( double * ) malloc( n_A[ i ] * sizeof( double ) );
/*	
	// begin test
	double test_time;
    double * mat_col;
    int n = 100000;

    mat_col = ( double * ) malloc( n*20000* sizeof(double) );

	A_fp = fopen(A_hdd_fname,"r");

	clock_gettime(CLOCK_MONOTONIC, & t1 );

    for ( i=0;i <5; i++ ) {
	  fread( mat_col, sizeof(double), n*20000, A_fp );
    }

	clock_gettime( CLOCK_MONOTONIC, & t2 );
	diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
	test_time = ( double ) diff / (1E9);

    printf("test time is %.5e \n", test_time);

    fclose(A_fp);
    free( mat_col );
	// end test
*/

	// Generate matrix.
	matrix_generate_ooc( n_A[ i ], n_A[ i ], A_ssd_fname );
	matrix_generate_ooc( n_A[ i ], n_A[ i ], A_hdd_fname );

    if ( n_A[ i ] <= 45000 ) {
	  A_fp = fopen( A_ssd_fname, "r" );
	}

	for ( j=0; j < n_A[i]; j++ ) {
	  buff_p[ j ] = j;
	}	
	
	// Factorize matrix.
	printf( "%% Working on n = %d \n", n_A[ i ] );

		// start timing
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do SSD factorization
		hqrrp_ooc( A_ssd_fname, n_A[i], n_A[i], n_A[i], buff_p, buff_tau,
					bl_size, k, p, 1 );

		// stop timing and record time
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_cpqr_ssd[ i ] = ( double ) diff / (1E9);
		
		// start timing
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do HDD factorization
		hqrrp_ooc( A_hdd_fname, n_A[i], n_A[i], n_A[i], buff_p, buff_tau,
		  			bl_size, k, p, 1 );

		// stop timing and record time
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_cpqr_hdd[ i ] = ( double ) diff / (1E9);
		
		// start timing
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do in-core factorization
		if ( n_A[ i ] <= 45000 ) {
		  NoFLA_HQRRP_WY_blk_var4( n_A[i], n_A[i], buff_A, n_A[i], buff_p, buff_tau,
		   			bl_size, k, p, 1 );
		}

		// stop timing and record time
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_cpqr_in[ i ] = ( double ) diff / (1E9);

	// Free matrices and vectors.
	if ( n_A[ i ] <= 45000 ) {
	  free( buff_A );
	}
	free( buff_p );
    free( buff_tau );

    // remove file that stored matrix
	remove( A_ssd_fname );
	remove( A_hdd_fname );
    
  }

  // write results to file
  ofp = fopen( "cpqr_ooc_partial_times.m", & mode );

  fprintf( ofp, "%% block size was %d \n \n", bl_size );

	// write out vector of values of n used for these tests
	fprintf( ofp, "n_cpqr_ooc = [ \n" );

	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp,  "%d ", n_A[ i ] );
	}

	fprintf( ofp, "]; \n \n");

    // write out vector of times for SSD computation
	
	fprintf( ofp, "t_cpqr_ssd = [ \n" );

	for ( i=0; i < (sizeof(n_A) / sizeof(int)); i++ ) {
	  fprintf( ofp,  "%.2e ", t_cpqr_ssd[ i ] );
	}

	fprintf( ofp, "]; \n \n");
    
	// write out vector of times for HDD computation

	fprintf( ofp, "t_cpqr_hdd = [ \n" );

	for ( i=0; i < (sizeof(n_A) / sizeof(int)); i++ ) {
	  fprintf( ofp,  "%.2e ", t_cpqr_hdd[ i ] );
	}

	fprintf( ofp, "]; \n \n");
	
	// write out vector of times for HDD computation

	fprintf( ofp, "t_cpqr_in = [ \n" );

	for ( i=0; i < (sizeof(n_A) / sizeof(int)); i++ ) {
	  fprintf( ofp,  "%.2e ", t_cpqr_in[ i ] );
	}

	fprintf( ofp, "]; \n \n");

  fclose(ofp);

  printf( "%% End of Program\n" );

  return 0;
  
}

// ============================================================================
static void matrix_generate_ooc( int m_A, int n_A, char * A_fname ) {
  // populate the empty file pointed to by A_fp with a matrix
  // with random values 

  FILE * A_fp;
  double * col_p; // for storing one col at a time before transferring to disk
  int i,j;
  size_t err_check;


  A_fp = fopen( A_fname, "w" );
  col_p = ( double * ) malloc( m_A * sizeof( double ) );

  srand( 10 );

  // create matrix one col at a time and write to disk
  for ( j=0; j < n_A; j++ ) {
    for ( i=0; i < m_A; i++ ) {
	  col_p[ i ] = ( double ) rand() / ( double ) RAND_MAX; 
	}
	err_check = fwrite( col_p, sizeof( double ), m_A , A_fp );
      if ( err_check != m_A ) {
	    printf("Error! Write to disk failed \n");
	  }
  }

  fclose( A_fp );

  // free memory
  free(col_p);

}
