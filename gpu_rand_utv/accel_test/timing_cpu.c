#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include <mkl.h>

#include "NoFLA_UTV_WY_blk_var2.h"

#define l_max( a, b ) ( (a) >  (b) ? (a) : (b) )
#define l_min( a, b ) ( (a) < (b) ? (a) : (b) )


// ===================================================================
// Declaration of local prototypes

static void matrix_generate(int m_A, int n_A, double * buff_A, int ldim_A);

static void print_double_matrix(char * name, int m_A, int n_A,
		double * buff_A, int ldim_A);

static int Normal_random_matrix( int m_A, int n_A,
               double * buff_A, int ldim_A ); 

static double Normal_random_number( double mu, double sigma ); 

static struct timespec start_timer( void ); 

static double stop_timer( struct timespec t1 ); 
// ===================================================================

int main( int argc, char *argv[] ) {
  int	ldim_A;
  int   info;
  int lwork = -1;
  double work_opt;
  int   i, j;
  double * buff_A, * buff_tau, * buff_work;
  double * buff_ss, * buff_U, * buff_V;
  int * buff_jpvt;
  double * buff_B, * buff_C;
  int * buff_iwork;
  char all = 'A', t = 'T', n = 'N';
  double d_one = 1.0, d_zero = 0.0;

  int bl_size = 128;
  int n_A[] = {15000};//{500,1000,2000,3000,4000,5000,6000,8000,10000,12000,15000};
  int q = 2;
  int p = 0;

  // for timing
  struct timespec time1;
  double t_dgeqrf_cpu[ sizeof( n_A ) / sizeof( int ) ];
  double t_dgesdd_cpu[ sizeof( n_A ) / sizeof( int ) ];
  double t_dgeqp3_cpu[ sizeof( n_A ) / sizeof( int ) ];
  double t_randutv_cpu[ sizeof( n_A ) / sizeof( int ) ];
  double t_dgemm_cpu[ sizeof( n_A ) / sizeof( int ) ];

  // for output file
  FILE * ofp;
  char mode = 'a';

  // begin loop
  for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) { 
    printf( "Working on n = %d \n", n_A[ i ] );

    // memory allocation for this loop
	buff_A = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof(double) );
	buff_tau = ( double * ) malloc( l_min( n_A[ i ], n_A[ i ] ) * 
									sizeof( double ) );
	buff_jpvt = ( int * ) malloc( n_A[ i ] * sizeof( int ) );

	buff_U = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );
	buff_V = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof( double ) );
	buff_ss = ( double * ) malloc( n_A[ i ] * sizeof( double ) );
	
	buff_B = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof(double) );
	buff_C = ( double * ) malloc( n_A[ i ] * n_A[ i ] * sizeof(double) );

	ldim_A = l_max( 1, n_A[ i ] );

	// dgeqrf

	  //generate matrix A
	  Normal_random_matrix( n_A[ i ], n_A[ i ], buff_A, ldim_A );

	  // get optimal size of work array
	  lwork = -1;
	  dgeqrf_( & n_A[ i ], & n_A[ i ], buff_A, & ldim_A,
			   buff_tau,
			   & work_opt, & lwork, & info );
	  lwork = ( int ) work_opt;
	  buff_work = ( double * ) malloc( lwork * sizeof(double) );

	  // compute unpivoted QR of A
	  time1 = start_timer(); 
	  dgeqrf( & n_A[ i ], & n_A[ i ], 
			  buff_A, & ldim_A,
			  buff_tau, 
			  buff_work, & lwork, & info );
	  if ( info != 0 ) {
		printf( "Error! dgeqrf failed!" );
		return 1;
	  }
	  
	  // compute matrix Q
	  lwork = -1;
	  dorgqr( & n_A[ i ], & n_A[ i ], & n_A[ i ],
			  buff_A, & ldim_A,
			  buff_tau,
			  & work_opt, & lwork, & info );
	  lwork = ( int ) work_opt;
	  free( buff_work ); 
	  buff_work = ( double * ) malloc( lwork * sizeof( double ) );
	  dorgqr( & n_A[ i ], & n_A[ i ], & n_A[ i ],
			  buff_A, & ldim_A,
			  buff_tau,
			  buff_work, & lwork, & info);
	  if ( info != 0 ) {
		printf( "Error! dorgqr failed!" );
		return 1;
	  }

	  t_dgeqrf_cpu[ i ] = stop_timer(time1);
	
	// dgesdd

	  //generate matrix A
	  Normal_random_matrix( n_A[ i ], n_A[ i ], buff_A, ldim_A );

	  // get work array info
	  buff_iwork = ( int * ) malloc( 8 * n_A[ i ] * sizeof( int ) );
	  lwork = -1;
	  dgesdd( & all, & n_A[ i ], & n_A[ i ],
			  buff_A, & ldim_A,
			  buff_ss, buff_U, & ldim_A, buff_V, & ldim_A,
			  & work_opt, & lwork, buff_iwork, & info );	
	  lwork = ( int ) work_opt;
	  free( buff_work );
	  buff_work = ( double * ) malloc( lwork * sizeof( double ) );

	  // perform factorization
	  time1 = start_timer(); 
	  dgesdd( & all, & n_A[ i ], & n_A[ i ],
			  buff_A, & ldim_A,
			  buff_ss, buff_U, & ldim_A, buff_V, & ldim_A,
			  buff_work, & lwork, buff_iwork, & info );	
			  // note: dgesdd actually returns Vt, not V;
			  //	   irrelevant here
	  if ( info != 0 ) {
		printf( "Error! dgesdd failed!" );
		return 1;
	  }
	  t_dgesdd_cpu[ i ] = stop_timer(time1);

	// dgeqp3

	  //generate matrix A
	  Normal_random_matrix( n_A[ i ], n_A[ i ], buff_A, ldim_A );

	  // initialize pivot vector
	  for ( j=0; j<n_A[ i ]; j++ ) {
	    buff_jpvt[ j ] = 0;
	  }

	  // get optimal size of work array
	  lwork = -1;
	  dgeqp3( & n_A[ i ], & n_A[ i ], buff_A, & ldim_A,
			   buff_jpvt, buff_tau,
			   & work_opt, & lwork, & info );
	  lwork = ( int ) work_opt;
	  free( buff_work );
	  buff_work = ( double * ) malloc( lwork * sizeof(double) );

	  // perform factorization
	  time1 = start_timer(); 
	  dgeqp3( & n_A[ i ], & n_A[ i ], buff_A, & ldim_A,
			   buff_jpvt, buff_tau,
			   buff_work, & lwork, & info );
	  if ( info != 0 ) {
		printf( "Error! dgeqp3 failed!" );
		return 1;
	  }
	  
	  // form Q matrix
	  lwork = -1;
	  dorgqr( & n_A[ i ], & n_A[ i ], & n_A[ i ],
			  buff_A, & ldim_A,
			  buff_tau,
			  & work_opt, & lwork, & info );
	  lwork = ( int ) work_opt;
	  free( buff_work ); 
	  buff_work = ( double * ) malloc( lwork * sizeof( double ) );
	  dorgqr( & n_A[ i ], & n_A[ i ], & n_A[ i ],
			  buff_A, & ldim_A,
			  buff_tau,
			  buff_work, & lwork, & info);
	  if ( info != 0 ) {
		printf( "Error! dorgqr failed!" );
		return 1;
	  }

	  t_dgeqp3_cpu[ i ] = stop_timer(time1);

	// randutv

	  //generate matrix A
	  Normal_random_matrix( n_A[ i ], n_A[ i ], buff_A, ldim_A );

	  // perform factorization
	  time1 = start_timer(); 
	  NoFLA_UTV_WY_blk_var2( n_A[ i ], n_A[ i ], buff_A, ldim_A,
							 1, n_A[ i ], n_A[ i ], buff_U, ldim_A,
							 1, n_A[ i ], n_A[ i ], buff_V, ldim_A,
							 bl_size, p, q );
	  t_randutv_cpu[ i ] = stop_timer(time1);

	// dgemm

	  // generate matrices A, B, C
	  Normal_random_matrix( n_A[ i ], n_A[ i ], buff_A, ldim_A );
	  Normal_random_matrix( n_A[ i ], n_A[ i ], buff_B, ldim_A );
	  Normal_random_matrix( n_A[ i ], n_A[ i ], buff_C, ldim_A );

	  // perform operation
	  time1 = start_timer(); 
	  dgemm( & n, & n, & n_A[ i ], & n_A[ i ], & n_A[ i ],
			 & d_one, 
			 buff_A, & ldim_A, buff_B, & ldim_A,
			 & d_zero,
			 buff_C, & ldim_A );
	  t_dgemm_cpu[ i ] = stop_timer(time1);

	// free matrices
	free( buff_A );
	free( buff_tau );
	free( buff_jpvt );

	free( buff_U );
	free( buff_V );
	free( buff_ss );

	free( buff_B );
	free( buff_C );

	free( buff_work );
    free( buff_iwork );

  } // for i

  // write results to file
  ofp = fopen( "times_cpu.m", & mode );

	fprintf( ofp, "%% block size for randUTV was %d \n", bl_size );
	fprintf( ofp, "%% q value for randUTV was %d \n", q );
	fprintf( ofp, "%% p value for randUTV was %d \n", p );

	// write out vector of values of n used for these tests
	fprintf( ofp, "n_rutv_cpu = [ \n" );
	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp, "%d ", n_A[ i ] );
	}
	fprintf( ofp, "]; \n \n" );

	// write out times for dgeqrf
	fprintf( ofp, "t_dgeqrf_cpu = [ \n" );
	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp, "%.2e ", t_dgeqrf_cpu[ i ] );
	}
	fprintf( ofp, "]; \n \n" );

	// write out times for dgesdd
	fprintf( ofp, "t_dgesdd_cpu = [ \n" );
	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp, "%.2e ", t_dgesdd_cpu[ i ] );
	}
	fprintf( ofp, "]; \n \n" );

	// write out times for dgeqp3
	fprintf( ofp, "t_dgeqp3_cpu = [ \n" );
	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp, "%.2e ", t_dgeqp3_cpu[ i ] );
	}
	fprintf( ofp, "]; \n \n" );

	// write out times for randutv
	fprintf( ofp, "t_randutv_cpu = [ \n" );
	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp, "%.2e ", t_randutv_cpu[ i ] );
	}
	fprintf( ofp, "]; \n \n" );

	// write out times for dgemm
	fprintf( ofp, "t_dgemm_cpu = [ \n" );
	for ( i=0; i < sizeof( n_A ) / sizeof( int ); i++ ) {
	  fprintf( ofp, "%.2e ", t_dgemm_cpu[ i ] );
	}
	fprintf( ofp, "]; \n \n" );

  fclose( ofp );

  printf("%% End of Program\n");

  return 0;
}

// ===============================================================================
static void matrix_generate(int m_A, int n_A, double * buff_A, int ldim_A) {
  int i,j;

  srand(10);
  for (j = 0; j < n_A; j++) {
    for (i=0; i < m_A; i++) {
      buff_A[i + j * ldim_A] = (double) rand() / (double) RAND_MAX;
    }
  }
}

// ===============================================================================
static void print_double_matrix(char * name, int m_A, int n_A, double * buff_A, int ldim_A) {
  int i,j;

  printf( "%s = [\n",name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%le ", buff_A[ i + j * ldim_A ] );
    }
    printf( "\n" );
  }
  printf( "];\n" );
}

// ============================================================================
static int Normal_random_matrix( int m_A, int n_A,
               double * buff_A, int ldim_A ) {
//
// It generates a random matrix with normal distribution.
//
  int  i, j;

  // Main loop.
  for ( j = 0; j < n_A; j++ ) {
    for ( i = 0; i < m_A; i++ ) {
      buff_A[ i + j * ldim_A ] = Normal_random_number( 0.0, 1.0 );
    }
  }

  return 0;
}

// ============================================================================
static double Normal_random_number( double mu, double sigma ) {
  static int     alternate_calls = 0;
  static double  b1, b2;
  double         c1, c2, a, factor;

  // Quick return.
  if( alternate_calls == 1 ) {
    alternate_calls = ! alternate_calls;
    return( mu + sigma * b2 );
  }
  // Main loop.
  do {
    c1 = -1.0 + 2.0 * ( (double) rand() / RAND_MAX );
    c2 = -1.0 + 2.0 * ( (double) rand() / RAND_MAX );
    a = c1 * c1 + c2 * c2;
  } while ( ( a == 0 )||( a >= 1 ) );
  factor = sqrt( ( -2 * log( a ) ) / a );
  b1 = c1 * factor;
  b2 = c2 * factor;
  alternate_calls = ! alternate_calls;
  return( mu + sigma * b1 );
}

// ======================================================================== 
static struct timespec start_timer( void ) { 
  // this function returns a timespec object that contains
  // clock information at the time of this function's execution
  //
  // performs the same function as MATLAB's 'tic'
 
  // declare variables
  struct timespec t1;

  // get current clock info
  clock_gettime( CLOCK_MONOTONIC, & t1 );

  return t1;

}
	
// ======================================================================== 
static double stop_timer( struct timespec t1 ) {
  // this function returns a variable of type double that
  // corresponds to the number of seconds that have elapsed
  // since the time that t1 was generated by start_timer
  // 
  // performs the same function as MATLAB's 'toc'
  //
  // t1: the output of start_timer; holds clock information
  //     from a function call to start_timer
  
  // declare variables 
  struct timespec  t2;
  uint64_t  t_elapsed_nsec;
  double    t_elapsed_sec;

  // get current clock info
  clock_gettime(CLOCK_MONOTONIC, & t2);

  // calculate elapsed time
  t_elapsed_nsec = (1000000000L) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
  t_elapsed_sec = (double) t_elapsed_nsec / (1000000000L);

  return t_elapsed_sec;

}
