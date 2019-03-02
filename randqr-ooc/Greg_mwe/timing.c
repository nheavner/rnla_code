#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <mkl.h>

#include <time.h>
#include "hqrrp_ooc.h"
#include "NoFLA_HQRRP_WY_blk_var4.h"

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )



// ============================================================================
// Declaration of local prototypes.

static void matrix_generate_ooc( int m_A, int n_A, char * dir_name, size_t dir_name_size,
								 char * A_fname, size_t A_fname_size );

// ============================================================================
int main() {
  
  int     ldim_A;
  double  * buff_A, * buff_tau;
  int  * buff_p;
  char dir_name_ssd[] = "./";
  char A_fname_ssd[] = "A_mat_ssd";

  char file_path_ssd[ sizeof( dir_name_ssd ) / sizeof( dir_name_ssd[0] ) + 
		sizeof( A_fname_ssd ) / sizeof( A_fname_ssd[0] ) ];
  strcpy( file_path_ssd, dir_name_ssd );
  strcat( file_path_ssd, A_fname_ssd );

  size_t read_check;

  int i, j;

  int bl_size = 250;
  int p = 0;
  int n_A[] = {1000,2000,4000, 10000, 20000};//{1000, 2000, 4000, 5000, 8000, 10000, 15000, 20000, 30000, 40000, 45000, 50000, 70000, 100000, 110000, 120000, 150000};


  // for timing
  struct timespec t1, t2;
  uint64_t diff;
  double   t_cpqr_single[ (sizeof( n_A ) / sizeof( int )) ];
  double   t_cpqr_multi[ (sizeof( n_A ) / sizeof( int )) ];

  // for output file
  FILE * ofp, * A_fp;
  char mode = 'a';

  for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {

	// Create matrix A, matrix P, and vector tau.
	buff_p    = ( int * )  malloc( n_A[ i ] * sizeof( int ) ); 

	buff_tau  = ( double * ) malloc( n_A[ i ] * sizeof( double ) );

	// Generate matrix.
	matrix_generate_ooc( n_A[ i ], n_A[ i ], dir_name_ssd, sizeof( dir_name_ssd ),
						 A_fname_ssd, sizeof( A_fname_ssd ) );

	for ( j=0; j < n_A[i]; j++ ) {
	  buff_p[ j ] = j;
	}	
	
	// Factorize matrix.
	printf( "%% Working on n = %d \n", n_A[ i ] );

		// start timing
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do factorization with NO multithreading
		hqrrp_ooc_physical_pivot( dir_name_ssd, sizeof( dir_name_ssd ), 
					A_fname_ssd, sizeof( dir_name_ssd ),
					n_A[i], n_A[i], n_A[i], buff_p, buff_tau,
					bl_size, n_A[i], p, 1 );

		// stop timing and record time
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_cpqr_single[ i ] = ( double ) diff / (1E9);

		// re-generate input matrix
		matrix_generate_ooc( n_A[ i ], n_A[ i ], dir_name_ssd, sizeof( dir_name_ssd ),
							 A_fname_ssd, sizeof( A_fname_ssd ) );

		for ( j=0; j < n_A[i]; j++ ) {
		  buff_p[ j ] = j;
		}	

		// start timing
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do factorization WITH multithreading
		hqrrp_ooc_multithreaded( dir_name_ssd, sizeof( dir_name_ssd ), 
					A_fname_ssd, sizeof( dir_name_ssd ),
					n_A[i], n_A[i], n_A[i], buff_p, buff_tau,
					bl_size, n_A[i], p, 1 );

		// stop timing and record time
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_cpqr_multi[ i ] = ( double ) diff / (1E9);

	// Free matrices and vectors.
	free( buff_p );
    free( buff_tau );

    // remove file that stored matrix
	remove( file_path_ssd );
    
  }

  // write results to file
  ofp = fopen( "cpqr_ooc_times.m", & mode );

  fprintf( ofp, "%% block size was %d \n \n", bl_size );

	// write out vector of values of n used for these tests
	fprintf( ofp, "n_cpqr_ooc = [ \n" );

	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp,  "%d ", n_A[ i ] );
	}

	fprintf( ofp, "]; \n \n");

    // write out vector of times for single-thread computation
	
	fprintf( ofp, "t_cpqr_single = [ \n" );

	for ( i=0; i < (sizeof(n_A) / sizeof(int)); i++ ) {
	  fprintf( ofp,  "%.2e ", t_cpqr_single[ i ] );
	}

	fprintf( ofp, "]; \n \n");

    // write out vector of times for multi-thread computation
	
	fprintf( ofp, "t_cpqr_multi = [ \n" );

	for ( i=0; i < (sizeof(n_A) / sizeof(int)); i++ ) {
	  fprintf( ofp,  "%.2e ", t_cpqr_multi[ i ] );
	}

	fprintf( ofp, "]; \n \n");

  fclose(ofp);

  printf( "%% End of Program\n" );

  return 0;
  
}

// ============================================================================
static void matrix_generate_ooc( int m_A, int n_A, char * dir_name, size_t dir_name_size,
								 char * A_fname, size_t A_fname_size ) {
  // populate the empty file pointed to by A_fp with a matrix
  // with random values 

  FILE * A_fp;
  double * col_p; // for storing one col at a time before transferring to disk
  int i,j;
  size_t err_check;

  char file_path[ dir_name_size / sizeof( dir_name[0] ) + 
		A_fname_size / sizeof( A_fname[0] ) ];
  strcpy( file_path, dir_name );
  strcat( file_path, A_fname );
  

  A_fp = fopen( file_path, "w" );
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
