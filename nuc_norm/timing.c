#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <mkl.h>
#include "compute_nuc_norm.h"

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )

//#define COMPUTE_SVD

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
  int i, j;
  double upper_T_norm = 0.0;

  int n_arr[] = {500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000};

  //for timing
  double t_proc;
  struct timespec ts_start, ts_end;

  //for output file
  FILE * ofp;
  char * mode = "a";

#ifdef COMPUTE_SVD
  //for computing SVD
  double * buff_Acp, * buff_ss, * buff_work;
  MKL_INT lwork, info;
  double svd_times[ sizeof( n_arr ) / sizeof( int ) ];
#endif

  ofp = fopen( "times.txt", mode );

  //fprintf( ofp, "values of n used in timing: 500,1000,2000,3000,4000,5000,6000,8000,10000 \n" );

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

#ifdef COMPUTE_SVD
    buff_Acp = ( double * ) malloc( m_A * n_A * sizeof( double ) );
    buff_ss = ( double * ) malloc( min(m_A, n_A) * sizeof( double ) );
    lwork = max( 3 * min( m_A, n_A ) + max( m_A, n_A ), 5 * min( m_A, n_A ) );
    buff_work = ( double * ) malloc( lwork * sizeof( double ) );
#endif

    // Generate matrix.
    matrix_generate( m_A, n_A, buff_A, ldim_A );

#ifdef COMPUTE_SVD
    //copy matrix
    for ( j=0; j < m_A*n_A; j++ ){
      buff_Acp[ j ] = buff_A[ j ];
    }
#endif

    // Factorize matrix.
    printf( "%% Just before computing factorization.\n" );
    // New factorization.
    
    clock_gettime( CLOCK_MONOTONIC, &ts_start );

	compute_nuc_norm( m_A, n_A, buff_A, ldim_A, 
        0, m_A, m_A, buff_U, ldim_U, 
        0, n_A, n_A, buff_V, ldim_V, 
		& upper_T_norm,
		64, 0, 2 );
   
    clock_gettime( CLOCK_MONOTONIC, &ts_end );
    t_proc = (double) ( (ts_end.tv_sec - ts_start.tv_sec)
            + (ts_end.tv_nsec - ts_start.tv_nsec) / (1E9) );
	fprintf( ofp, "%.2e ", t_proc );
    printf( "%% Just after computing factorization.\n" );

#ifdef COMPUTE_SVD
    printf( "%% Beginning SVD factorization.\n" );
    clock_gettime( CLOCK_MONOTONIC, &ts_start );

    dgesvd( & n, & n, & m_A, & n_A,
            buff_Acp, & ldim_A, buff_ss,
            NULL, & m_A, NULL, & n_A,
            buff_work, & lwork, & info );

    clock_gettime( CLOCK_MONOTONIC, &ts_end );
    printf( "%%Done with SVD factorization; info = %i \n", info );

    svd_times[ i ] = (double) ( (ts_end.tv_sec - ts_start.tv_sec)
                    + (ts_end.tv_nsec - ts_start.tv_nsec) / (1E9) );

#endif

    // Free matrices and vectors.
    free( buff_A );
    free( buff_U );
    free( buff_V );

#ifdef COMPUTE_SVD
    free( buff_Acp );
    free( buff_ss );
    free( buff_work );
#endif

  }
 
  fprintf( ofp, "\n" );

#ifdef COMPUTE_SVD
  fprintf( ofp, "SVD times: " );
  for ( i=0; i < (sizeof(svd_times)/sizeof(double)); i++ ){
    fprintf( ofp, "%.2e ", svd_times[ i ] );
  }
  fprintf( ofp, "\n" );
#endif

  fclose( ofp );

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

