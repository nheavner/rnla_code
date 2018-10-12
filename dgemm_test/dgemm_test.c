/*
how to compile:
gcc -c dgemm_test.c
gcc -o dgemm_test.x dgemm_test.o -Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -ldl -lpthread -lm -lgomp
*/

#include <math.h>
#include <stdlib.h>
#include<stdio.h>
#include<stdint.h>
#include <time.h>

#include <mkl.h>

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
  int64_t	nb_alg, pp, m_A, n_A, ldim_A, m_B, n_B, ldim_B, ldim_C;
  int   i, j;
  double * buff_A, * buff_B, * buff_C;
  char all = 'A', t = 'T', n = 'N';
  double d_one = 1.0, d_zero = 0.0;
  struct timespec time1;
  double t_dgemm = 0.0;

  // create matrices A, B, C
  m_A = 10000;
  n_A = 10000;
  ldim_A = l_max(1,m_A);
  m_B = 10000;
  n_B = 10000;
  ldim_B = l_max(1,m_B);
  ldim_C = ldim_A;

  buff_A = ( double * ) malloc( m_A * n_A * sizeof(double) );
  buff_B = ( double * ) malloc( m_B * n_B * sizeof(double) );
  buff_C = ( double * ) malloc( m_A * n_B * sizeof(double) );

  //generate matrices A,B
  Normal_random_matrix( m_A, n_A, buff_A, ldim_A );
  Normal_random_matrix( m_B, n_B, buff_B, ldim_B );

  //print_double_matrix("A",m_A,n_A,buff_A,m_A);
  //print_double_matrix("B",m_B,n_B,buff_B,m_B);

  time1 = start_timer(); 

  for (i=0; i < 10; i++) {
	//compute C = A * B
	dgemm_( & n, & n, & m_A, & n_B, & n_A, 
		& d_one, buff_A, & ldim_A,
			 buff_B, & ldim_B,
		& d_zero,buff_C, & ldim_C);
  }

  t_dgemm += stop_timer(time1);
  t_dgemm = t_dgemm / 10;

  printf("t_dgemm = %.2e \n", t_dgemm);

  //print results
  //printf("Multiplication complete\n");
  //print_double_matrix("C",m_A,n_B,buff_C,m_A);

  // free matrices
  free(buff_A);
  free(buff_B);
  free(buff_C);

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
