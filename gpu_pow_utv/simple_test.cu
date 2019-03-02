#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "pow_utv_gpu.h"
#include <mkl.h>

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )

//#define PRINT_DATA
#define CHECK_ERROR

// ============================================================================
// Declaration of local prototypes.

static void matrix_generate( int m_A, int n_A, 
				double * buff_A, int ldim_A );

#ifdef PRINT_DATA
static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A );
#endif

// ============================================================================
int main() {
  int     q_iter, 
		  n_A, ldim_A, 
		  ldim_U, ldim_V;
  double  * buff_A, * buff_U, * buff_V;
 
#ifdef PRINT_DATA
  char A_name[] = "A", U_name[] = "U1", 
	   V_name[] = "V1", T_name[] = "T1"; 

  char * A_name_pt = A_name, * U_name_pt = U_name, 
	   * V_name_pt = V_name, * T_name_pt = T_name;
#endif

#ifdef CHECK_ERROR
  char f = 'f', n = 'n', t = 't';
  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0; 
  
  double * buff_UT, * buff_Ac, * buff_Acc;
  
  double err, norm_A;
  double * buff_ss;

  double * buff_work;
  int    * buff_iwork;
  int    lwork;
  int    info;

  int    i;
#endif


  // Create matrix A, matrix U, and matrix V.
  n_A      = 1000;
  q_iter = 1;

  buff_A   = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  ldim_A   = max( 1, n_A );

  buff_U   = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  ldim_U   = max( 1, n_A );

  buff_V   = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  ldim_V   = max( 1, n_A );

#ifdef CHECK_ERROR
  buff_Ac  = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  buff_Acc = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  buff_UT  = ( double * ) malloc( n_A * n_A * sizeof( double ) );

  buff_ss  = ( double * ) malloc( n_A * sizeof( double ) );

  // allocate memory for work arrays
  lwork = 3 * n_A + 7 * n_A; 

  buff_work = ( double * ) malloc( lwork * sizeof( double ) );
  buff_iwork = ( int * ) malloc( 8 * n_A * sizeof( int ) );
#endif

  // Generate matrix.
  matrix_generate( n_A, n_A, buff_A, ldim_A );

#ifdef CHECK_ERROR
  // copy data to later check error
  for ( i=0; i < n_A * n_A; i++ ) {
    buff_Ac[ i ] = buff_A[ i ];
  }
  
  for ( i=0; i < n_A * n_A; i++ ) {
    buff_Acc[ i ] = buff_A[ i ];
  }
#endif

#ifdef PRINT_DATA
  printf("q = %d; \n", q_iter );
  print_double_matrix( A_name_pt, n_A, n_A, buff_A, ldim_A );
#endif

  // Factorize matrix.
  printf( "%% Just before computing factorization.\n" );

  pow_utv_gpu( n_A, n_A, buff_A, ldim_A, 
      1, n_A, n_A, buff_U, ldim_U, 
      1, n_A, n_A, buff_V, ldim_V, 
      q_iter );

  printf( "%% Just after computing factorization.\n" );

  // Print results.
#ifdef PRINT_DATA
  print_double_matrix( T_name_pt, n_A, n_A, buff_A, ldim_A );
  print_double_matrix( U_name_pt, n_A, n_A, buff_U, ldim_U );
  print_double_matrix( V_name_pt, n_A, n_A, buff_V, ldim_V );
#endif

  // check backward error

#ifdef CHECK_ERROR

  // compute || A ||
  norm_A = dlange(  & f,  & n_A,  & n_A, 
				buff_Ac,  & n_A, NULL );

  // compute U * T
  dgemm( & n, & n, & n_A, & n_A, & n_A,
		& d_one, buff_U, & n_A,
		buff_A, & ldim_A,
		& d_zero, buff_UT, & n_A );

  // compute (U * T) * V' - A, store in buff_Ac
  dgemm( & n, & t, & n_A, & n_A, & n_A,
		& d_one, buff_UT, & n_A,
		buff_V, & n_A,
		& d_neg_one, buff_Ac, & n_A );

  // compute || A - U * T * V'  ||
  err = dlange(  & f,  & n_A,  & n_A, 
				buff_Ac,  & n_A, NULL );

  printf("err = %.2e \n", err);

  printf( "%% || A - U * T * V' ||_F / || A ||_F = %e \n", err / norm_A );

  // check how far away T is from being diagonal

    // compute singular values of A
	dgesdd( & n, 
			& n_A, & n_A, buff_Acc, & ldim_A,
		    buff_ss,
			NULL, & n_A,
			NULL, & n_A,
			buff_work, & lwork, buff_iwork, & info );

    // compute SIGMA - T
    for ( i=0; i < n_A; i++ ) {
	  buff_A[ i + i * ldim_A ] = buff_A[ i + i * ldim_A ] - buff_ss[ i ];  
	}

	// compute || SIGMA - T ||
	err = dlange( & f, & n_A, & n_A,
			buff_A, & n_A, NULL );
	
  printf( "%% || D - T ||_F / || A ||_F = %e \n", err / norm_A );

  // compute the error in the singular value estimates vs the singular values
  err = 0.0;
  for ( i=0; i < n_A; i++ ) {
    err += buff_A[ i + i * ldim_A ] * buff_A[ i + i * ldim_A ]; 
  }
  err = sqrt( err );

  printf( "%% || diag(D) - diag(T) || = %e \n", err );

#endif

  // Free matrices and vectors.
  free( buff_A );
  free( buff_U );
  free( buff_V );
  
#ifdef CHECK_ERROR
  free( buff_Ac );
  free( buff_Acc );
  free( buff_UT );

  free( buff_ss );

  free( buff_work );
  free( buff_iwork );
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
#ifdef PRINT_DATA
static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A ) {
  int  i, j;

  printf( "%s = [\n", name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%.16e ", buff_A[ i + j * ldim_A ] );
    }
    printf( ";\n" );
  }
  printf( "];\n" );
}
#endif
