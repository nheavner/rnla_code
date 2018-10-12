#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <time.h>

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )

// ============================================================================
static void matrix_generate_ooc( int m_A, int n_A, char * A_fname );

// ============================================================================
int main( int argc, char *argv[] ) {
  int m_A, n_A;
  char    A_fname_ssd[] = "A_mat";
  char    A_fname_hdd[] = "/media/hdd/A_mat";
  double * A_p; // points to array that we will write to
  FILE * A_fp_ssd, * A_fp_hdd;
  int i;
  size_t read_check;

  struct timespec t1, t2, t_start, t_end;
  uint64_t diff;
  double t_read_ssd = 0.0, t_write_ssd = 0.0;
  double t_read_hdd = 0.0, t_write_hdd = 0.0;
  double t_total = 0.0;

  // Create matrix A, vector p, vector s, and matrix Q.
  m_A      = 45000;
  n_A      = 45000;
 
  clock_gettime( CLOCK_MONOTONIC, & t_start );

  A_p = ( double * ) malloc( m_A * n_A * sizeof( double ) );

  // Generate binary file which stores the matrix (out of core)
  matrix_generate_ooc( m_A, n_A, A_fname_ssd ); 
  matrix_generate_ooc( m_A, n_A, A_fname_hdd ); 

  printf( "The matrices have been generated \n" );

  // read matrix and time
  A_fp_ssd = fopen( A_fname_ssd, "r+" );
  A_fp_hdd = fopen( A_fname_hdd, "r+" );
 
	// read from ssd
	clock_gettime( CLOCK_MONOTONIC, & t1 );
	
	read_check = fread( A_p, sizeof( double ), m_A * n_A, A_fp_ssd );
   
	clock_gettime( CLOCK_MONOTONIC, & t2 );
	diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
	t_read_ssd += ( double ) diff / (1E9);

    fseek( A_fp_ssd, 0, SEEK_SET );
 
	// read from hdd
	clock_gettime( CLOCK_MONOTONIC, & t1 );
	
	read_check = fread( A_p, sizeof( double ), m_A * n_A, A_fp_hdd );
	
	clock_gettime( CLOCK_MONOTONIC, & t2 );
	diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
	t_read_hdd += ( double ) diff / (1E9);

    fseek( A_fp_hdd, 0, SEEK_SET );

  // write to hard drives and time
	
	// write to ssd
	clock_gettime( CLOCK_MONOTONIC, & t1 );

	read_check = fwrite( A_p, sizeof( double ), m_A * n_A, A_fp_ssd );

	clock_gettime( CLOCK_MONOTONIC, & t2 );
	diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
	t_write_ssd += ( double ) diff / (1E9);
	
	// write to hdd
	clock_gettime( CLOCK_MONOTONIC, & t1 );

	read_check = fwrite( A_p, sizeof( double ), m_A * n_A, A_fp_hdd );

	clock_gettime( CLOCK_MONOTONIC, & t2 );
	diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
	t_write_hdd += ( double ) diff / (1E9);

  // close file
  fclose( A_fp_ssd );
  fclose( A_fp_hdd );


  // print out time required for reading, writing
  printf( "Time required for reading from ssd: %le\n", t_read_ssd );
  printf( "Time required for reading from hdd: %le\n", t_read_hdd );
  printf( "Time required for writing to ssd: %le\n", t_write_ssd );
  printf( "Time required for writing to hdd: %le\n", t_write_hdd );

  // remove files that stored matrix
  remove( A_fname_ssd );
  remove( A_fname_hdd );

  // Free matrices and vectors.
  free( A_p );

  clock_gettime( CLOCK_MONOTONIC, & t_end );
  diff = (1E9) * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_nsec - t_start.tv_nsec;
  t_total += ( double ) diff / (1E9);

  printf( "Total time: %le\n", t_total );

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
