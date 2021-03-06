#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <mkl.h>
#include "rand_utv_ooc.h"

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )

// ============================================================================
// Declaration of local prototypes.

static void matrix_generate( int m_A, int n_A, double * buff_A, int ldim_A );

static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A );


// ============================================================================
int main( int argc, char *argv[] ) {
  
  MKL_INT     nb_alg, pp, m_A, n_A, mn_A, ldim_A, ldim_U, ldim_V;
  double  * buff_A, * buff_U, * buff_V;
  char all = 'A', t = 'T', n = 'N', f  = 'F';
  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0;
  int i;

  int q = 2;
  int bl_size = 256;
  int n_arr[] = {2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 18000};

  double t_proc;
  struct timespec ts_start, ts_end;
  
  for ( i=0; i < ( sizeof(n_arr) / sizeof(int) ); i++ ) {
    printf( "%% n = %i: \n", n_arr[ i ] );
	
	// Create matrix A, matrix U, and matrix V.
    m_A      = n_arr[ i ];
    n_A      = n_arr[ i ];
    mn_A     = min( m_A, n_A );

    buff_A   = ( double * ) malloc( m_A * n_A * sizeof( double ) );
    ldim_A   = max( 1, m_A );

    buff_U   = ( double * ) malloc( m_A * m_A * sizeof( double ) );
    ldim_U   = max( 1, m_A );

    buff_V   = ( double * ) malloc( n_A * n_A * sizeof( double ) );
    ldim_V   = max( 1, n_A );

    // Generate matrix.
    matrix_generate( m_A, n_A, buff_A, ldim_A );

    // Factorize matrix.
    // New factorization.
    // We use a small block size to factorize the small input matrix, but you
    // should use larger blocksizes such as 64 for larger matrices.
    
	rand_utv_ooc( m_A, n_A, buff_A, ldim_A, 
        0, m_A, m_A, buff_U, ldim_U, 
        0, n_A, n_A, buff_V, ldim_V, 
        bl_size, 0, q );

    // Free matrices and vectors.
    free( buff_A );
    free( buff_U );
    free( buff_V );
  }
 
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

// ============================================================================
static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A ) {
  int  i, j;

  printf( "%s = [\n", name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%le ", buff_A[ i + j * ldim_A ] );
    }
    printf( "\n" );
  }
  printf( "];\n" );
}

