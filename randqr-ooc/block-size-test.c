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
  double  * buff_tau;
  int  * buff_p;
  char A_ssd_fname[] = "./A_mat_ssd";

  size_t read_check;

  int i, j;

  int bl_size[] = {400,600};//{50,100,128,256,512,800,1024};
  int p = 0;

  int n_A = 70000;
  char n_A_str[6];
  sprintf(n_A_str, "%d", n_A );


  char of_fname[] = "./bl_size_test_data/bl_test_times_n_??k.m";
  of_fname[36] = n_A_str[0];
  of_fname[37] = n_A_str[1];

  // for timing
  struct timespec t1, t2;
  uint64_t diff;
  double   t_cpqr_ssd[ (sizeof( bl_size ) / sizeof( int )) ];

  // for output file
  FILE * ofp, * A_fp;
  char mode = 'a';

  // Create matrix A, matrix U, and matrix V.
  buff_p    = ( int * )  malloc( n_A * sizeof( int ) ); 

  buff_tau  = ( double * ) malloc( n_A * sizeof( double ) );

  
  for ( i=0; i < sizeof( bl_size ) / sizeof( int ); i++ ) {

	// Generate matrix.
	matrix_generate_ooc( n_A, n_A, A_ssd_fname );

	for ( j=0; j < n_A; j++ ) {
	  buff_p[ j ] = j;
	}	

    // Factorize matrix.
	printf( "%% Working on b = %d; %d/%d \n", bl_size[ i ], i+1, (int) ( sizeof( bl_size ) / sizeof( int ) ) );

		// start timing
		clock_gettime(CLOCK_MONOTONIC, & t1 );
		
		// do SSD factorization
		hqrrp_ooc( A_ssd_fname, n_A, n_A, n_A, buff_p, buff_tau,
					bl_size[ i ], p, 1 );

		// stop timing and record time
		clock_gettime( CLOCK_MONOTONIC, & t2 );
		diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
		t_cpqr_ssd[ i ] = ( double ) diff / (1E9);

  }

  // Free matrices and vectors.
  free( buff_p );
  free( buff_tau );

  // remove file that stored matrix
  remove( A_ssd_fname );

  // write results to file
  ofp = fopen( of_fname, & mode );

  fprintf( ofp, "%% matrix size was %d \n \n", n_A );

	// write out vector of values of n used for these tests
	fprintf( ofp, "bl_size = [ \n" );

	for ( i=0; i < sizeof( bl_size ) / sizeof( int ); i++ ) {
	  fprintf( ofp,  "%d ", bl_size[ i ] );
	}

	fprintf( ofp, "]; \n \n");

    // write out vector of times for SSD computation
	
	fprintf( ofp, "t_cpqr_ssd = [ \n" );

	for ( i=0; i < (sizeof(bl_size) / sizeof(int)); i++ ) {
	  fprintf( ofp,  "%.2e ", t_cpqr_ssd[ i ] );
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


  A_fp = fopen( A_fname, "w" );
  col_p = ( double * ) malloc( m_A * sizeof( double ) );

  srand( 10 );

  // create matrix one col at a time and write to disk
  for ( j=0; j < n_A; j++ ) {
    for ( i=0; i < m_A; i++ ) {
	  col_p[ i ] = ( double ) rand() / ( double ) RAND_MAX; 
	}
	fwrite( col_p, sizeof( double ), m_A , A_fp );
  }

  fclose( A_fp );

  // free memory
  free(col_p);

}
