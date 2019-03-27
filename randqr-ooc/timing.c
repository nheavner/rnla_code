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
  char dir_name_hdd[] = "/media/hdd/";
  char A_fname_ssd[] = "A_mat_ssd";
  char A_fname_hdd[] = "A_mat_hdd";

  char file_path_ssd[ sizeof( dir_name_ssd ) / sizeof( dir_name_ssd[0] ) + 
		sizeof( A_fname_ssd ) / sizeof( A_fname_ssd[0] ) ];
  strcpy( file_path_ssd, dir_name_ssd );
  strcat( file_path_ssd, A_fname_ssd );
  char file_path_hdd[ sizeof( dir_name_hdd ) / sizeof( dir_name_hdd[0] ) + 
		sizeof( A_fname_hdd ) / sizeof( A_fname_hdd[0] ) ];
  strcpy( file_path_hdd, dir_name_hdd );
  strcat( file_path_hdd, A_fname_hdd );

  size_t read_check;

  int i, j;

  int bl_size = 250;
  int k = 500;
  int num_cols_read = 250;//1000;
  int p = 0;
  int n_A[] = {1000, 8000, 15000, 30000, 40000, 50000, 70000, 100000, 120000, 150000};


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

	// Generate matrix.
	matrix_generate_ooc( n_A[ i ], n_A[ i ], dir_name_ssd, sizeof( dir_name_ssd ),
						 A_fname_ssd, sizeof( A_fname_ssd ) );
	matrix_generate_ooc( n_A[ i ], n_A[ i ], dir_name_hdd, sizeof( dir_name_hdd ), 
						 A_fname_hdd, sizeof( A_fname_hdd ) );

    // copy matrix to RAM
    if ( n_A[ i ] <= 45000 ) {
	  A_fp = fopen( file_path_ssd, "r" );
      read_check = fread( buff_A, sizeof( double ), n_A[ i ] * n_A[ i ], A_fp ); 
	  fclose( A_fp );
	}

	for ( j=0; j < n_A[i]; j++ ) {
	  buff_p[ j ] = j;
	}	
	
	// Factorize matrix.
	printf( "%% Working on n = %d \n", n_A[ i ] );

		// start timing
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do SSD factorization
		hqrrp_ooc( dir_name_ssd, sizeof( dir_name_ssd ), A_fname_ssd, sizeof( A_fname_ssd ),
					n_A[i], n_A[i], n_A[i], buff_p, buff_tau,
					bl_size, k, p, 1, num_cols_read );

		// stop timing and record time
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_cpqr_ssd[ i ] = ( double ) diff / (1E9);
		
		// start timing
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do HDD factorization
		hqrrp_ooc( dir_name_hdd, sizeof( dir_name_hdd ), A_fname_hdd, sizeof( A_fname_hdd ),
					n_A[i], n_A[i], n_A[i], buff_p, buff_tau,
		  			bl_size, k, p, 1, num_cols_read );

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
	remove( file_path_ssd );
	remove( file_path_hdd );
    
  }

  // write results to file
  ofp = fopen( "cpqr_ooc_partial_times_k500.m", & mode );

  fprintf( ofp, "%% block size was %d \n \n %% left lookking algorithm was used; k = %d \n", bl_size, k );

	// write out vector of values of n used for these tests
	fprintf( ofp, "n_cpqr_ooc_left = [ \n" );

	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp,  "%d ", n_A[ i ] );
	}

	fprintf( ofp, "]; \n \n");

    // write out vector of times for SSD computation
	
	fprintf( ofp, "t_cpqr_ssd_left = [ \n" );

	for ( i=0; i < (sizeof(n_A) / sizeof(int)); i++ ) {
	  fprintf( ofp,  "%.2e ", t_cpqr_ssd[ i ] );
	}

	fprintf( ofp, "]; \n \n");
    
	// write out vector of times for HDD computation

	fprintf( ofp, "t_cpqr_hdd_left = [ \n" );

	for ( i=0; i < (sizeof(n_A) / sizeof(int)); i++ ) {
	  fprintf( ofp,  "%.2e ", t_cpqr_hdd[ i ] );
	}

	fprintf( ofp, "]; \n \n");
	
	// write out vector of times for in core computation

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
