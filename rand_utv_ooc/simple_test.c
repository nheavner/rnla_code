#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "rand_utv_ooc.h"
#include <mkl.h>

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )

#define PRINT_DATA


// ============================================================================
// Declaration of local prototypes.

static void matrix_generate( int m_A, int n_A, 
				double * buff_A, int ldim_A );

static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A );


// ============================================================================
int main() {
  int     bl_size, pp, q_iter, m_A, n_A, mn_A, ldim_A, ldim_U, ldim_V;
  double  * buff_A, * buff_U, * buff_V, * buff_UT;
  double * buff_Ac;

  int i;
  double err, norm_A;
  char t = 'T', n = 'N', f = 'F';
  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0;

  // Create matrix A, matrix U, and matrix V.
  m_A      = 100;//100;
  n_A      = 100;//90;
  bl_size = 100;//8;
  q_iter = 1;
  mn_A     = min( m_A, n_A );

  buff_A   = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  ldim_A   = max( 1, m_A );

  buff_U   = ( double * ) malloc( m_A * m_A * sizeof( double ) );
  ldim_U   = max( 1, m_A );

  buff_V   = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  ldim_V   = max( 1, n_A );

  buff_Ac  = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  buff_UT  = ( double * ) malloc( m_A * n_A * sizeof( double ) );

  // Generate matrix.
  matrix_generate( m_A, n_A, buff_A, ldim_A );

  // copy data to later check error
  for ( i=0; i < m_A * n_A; i++ ) {
    buff_Ac[ i ] = buff_A[ i ];
  }

#ifdef PRINT_DATA
  print_double_matrix( "ai", m_A, n_A, buff_A, ldim_A );
#endif

  // Factorize matrix.
  printf( "%% Just before computing factorization.\n" );
  // New factorization.
  // We use a small block size to factorize the small input matrix, but you
  // should use larger blocksizes such as 64 for larger matrices.
  rand_utv_gpu( m_A, n_A, buff_A, ldim_A, 
      1, m_A, m_A, buff_U, ldim_U, 
      1, n_A, n_A, buff_V, ldim_V, 
      bl_size, 0, q_iter );
      //// 64, 10, 2 );
  printf( "%% Just after computing factorization.\n" );

  // Print results.
#ifdef PRINT_DATA
  print_double_matrix( "tf", m_A, n_A, buff_A, ldim_A );
  print_double_matrix( "uf", m_A, m_A, buff_U, ldim_U );
  print_double_matrix( "vf", n_A, n_A, buff_V, ldim_V );
#endif

  // check backward error
  
  // compute || A ||
  norm_A = dlange(  & f,  & m_A,  & n_A, 
				buff_Ac,  & m_A, NULL );

  // compute U * T
  dgemm( & n, & n, & m_A, & n_A, & m_A,
		& d_one, buff_U, & ldim_A,
		buff_A, & ldim_A,
		& d_zero, buff_UT, & m_A );

  // compute A - (U * T) * V', store in buff_Acp
  dgemm( & n, & t, & m_A, & n_A, & n_A,
		& d_one, buff_UT, & m_A,
		buff_V, & n_A,
		& d_neg_one, buff_Ac, & m_A );

  // compute || A - U * T * V'  ||
  err = dlange(  & f,  & m_A,  & n_A, 
				buff_Ac,  & m_A, NULL );

  printf( "%% || A - U * T * V' ||_F / || A ||_F = %e \n", err / norm_A );

  // Free matrices and vectors.
  free( buff_A );
  free( buff_U );
  free( buff_V );
  free( buff_Ac );
  free( buff_UT );

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

