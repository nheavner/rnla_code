#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <mkl.h>
#include "compute_nuc_norm.h"

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )

//#define PRINT_DATA
#define CHECK_ERROR


// ============================================================================
// Declaration of local prototypes.

static void matrix_generate( int m_A, int n_A, double * buff_A, int ldim_A );

static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A );


// ============================================================================
int main( int argc, char *argv[] ) {
  
  MKL_INT     nb_alg, pp, m_A, n_A, mn_A, ldim_A, ldim_U, ldim_V;
  double  * buff_A, * buff_Acp, * buff_Acpp, * buff_U, * buff_V, * buff_UT, * buff_ss;
  char all = 'A', t = 'T', n = 'N', f  = 'F';
  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0;
  double err;
  int i;
  double A_norm = 0.0, A_norm_est = 0.0, upper_T_norm = 0.0;

  // for dgesvd
  double * buff_work;
  MKL_INT lwork, info;

  // Create matrix A, matrix U, and matrix V.
  m_A      = 1000;
  n_A      = 1000;
  mn_A     = min( m_A, n_A );

  buff_A   = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  ldim_A   = max( 1, m_A );

  buff_U   = ( double * ) malloc( m_A * m_A * sizeof( double ) );
  ldim_U   = max( 1, m_A );

  buff_V   = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  ldim_V   = max( 1, n_A );

#ifdef CHECK_ERROR
  buff_Acp = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  buff_Acpp = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  buff_UT  = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  buff_ss  = ( double * ) malloc( min( m_A, n_A ) * sizeof( double ) );

  lwork = max( 3 * min( m_A, n_A ) + max( m_A, n_A ), 5 * min( m_A, n_A ) );
  buff_work = ( double * ) malloc( lwork * sizeof( double ) );
#endif

  // Generate matrix.
  matrix_generate( m_A, n_A, buff_A, ldim_A );

#ifdef CHECK_ERROR
  for ( i=0; i < m_A*n_A; i++ ) {
    buff_Acp[ i ] = buff_A[ i ];
	buff_Acpp[ i ] = buff_A[ i ];
  }
#endif

#ifdef PRINT_DATA
  print_double_matrix( "ai", m_A, n_A, buff_Acp, ldim_A );
#endif

  // Factorize matrix.
  printf( "%% Just before computing factorization.\n" );
  // New factorization.
  // We use a small block size to factorize the small input matrix, but you
  // should use larger blocksizes such as 64 for larger matrices.
  compute_nuc_norm( m_A, n_A, buff_A, ldim_A, 
      0, m_A, m_A, buff_U, ldim_U, 
      0, n_A, n_A, buff_V, ldim_V,
	  & upper_T_norm,
      2, 0, 2 );
      //// 64, 10, 2 );
  printf( "%% Just after computing factorization.\n" );

  // Print results.
#ifdef PRINT_DATA
  print_double_matrix( "af", m_A, n_A, buff_A, ldim_A );
  print_double_matrix( "uf", m_A, m_A, buff_U, ldim_U );
  print_double_matrix( "vf", n_A, n_A, buff_V, ldim_V );
#endif

#ifdef CHECK_ERROR
  // compute error in nuclear norm estimation
  
  // compute true nuclear norm
  dgesvd( & n, & n, & m_A, & n_A, 
		buff_Acp, & ldim_A, buff_ss, 
		NULL, & m_A, NULL, & n_A, 
		buff_work, & lwork, & info );

  for ( i=0; i < min( m_A, n_A ); i++ ) {
    A_norm += buff_ss[ i ];
	A_norm_est += buff_A[ i + i * ldim_A ];
  }

  printf( "( A_norm - A_norm_est ) / A_norm = %e \n", ( A_norm - A_norm_est ) / A_norm );

  // compute backward error

  // compute U * T
  dgemm_( & n, & n, & m_A, & n_A, & m_A,
		& d_one, buff_U, & ldim_A,
		buff_A, & ldim_A,
		& d_zero, buff_UT, & m_A );

  // compute A - (U * T) * V', store in buff_Acp
  dgemm_( & n, & t, & m_A, & n_A, & n_A,
		& d_one, buff_UT, & m_A,
		buff_V, & n_A,
		& d_neg_one, buff_Acpp, & m_A );

  // compute || A - U * T * V'  ||
  err = dlange(  & f,  & m_A,  & n_A, 
				buff_Acpp,  & m_A, NULL );

  printf( "|| A - U * T * V' || = %e \n", err );

  // print bound on estimate error
  printf( "|| U ||_F = %e \n", upper_T_norm );

#endif

  // Free matrices and vectors.
  free( buff_A );
  free( buff_U );
  free( buff_V );

#ifdef CHECK_ERROR
  free( buff_Acp );
  free( buff_UT );
  free( buff_ss );
  free( buff_work );
#endif

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
static void matrix_generate2( int m_A, int n_A, double * buff_A, int ldim_A ) {
  int  i, j, num;

  //
  // Matrix with integer values.
  // ---------------------------
  //
  if( ( m_A > 0 )&&( n_A > 0 ) ) {
    num = 1;
    for ( j = 0; j < n_A; j++ ) {
      for ( i = ( j % m_A ); i < m_A; i++ ) {
        buff_A[ i + j * ldim_A ] = ( double ) num;
        num++;
      }
      for ( i = 0; i < ( j % m_A ); i++ ) {
        buff_A[ i + j * ldim_A ] = ( double ) num;
        num++;
      }
    }
    if( ( m_A > 0 )&&( n_A > 0 ) ) {
      buff_A[ 0 + 0 * ldim_A ] = 1.2;
    }
#if 0
    // Scale down matrix.
    if( num == 0.0 ) {
      rnum = 1.0;
    } else {
      rnum = 1.0 / num;
    }
    for ( j = 0; j < n_A; j++ ) {
      for ( i = 0; i < m_A; i++ ) {
        buff_A[ i + j * ldim_A ] *= rnum;
      }
    }
#endif
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

