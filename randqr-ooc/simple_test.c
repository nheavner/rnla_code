#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <mkl.h>
#include "hqrrp_ooc.h"

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )

#define PRINT_DATA


// ============================================================================
// Declaration of local prototypes.

static void matrix_generate_ooc( int m_A, int n_A, char * A_fname );

static void matrix_generate( int m_A, int n_A, double * buff_A, int ldim_A );

static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A );

static void print_double_vector( char * name, int n, double * vector );

static void print_int_vector( char * name, int n, int * vector );

static void init_pvt( int n, int * vector );

static void set_pvt_to_zero( int n_p, int * buff_p );


// ============================================================================
int main( int argc, char *argv[] ) {
  int     nb_alg, pp, m_A, n_A, mn_A, ldim_A, ldim_Q, info, lwork;
  double  * buff_A, * buff_tau, * buff_Q, * buff_wk_qp4, * buff_wk_orgqr;
  int     * buff_p;
  FILE	  * A_fp; // pointer to the file that stores A
  char    A_fname[] = "A_mat";
  size_t  read_check;

  // Create matrix A, vector p, vector s, and matrix Q.
  m_A      = 10;
  n_A      = 8;
  nb_alg   = 3;
  pp	   = 0;
  mn_A     = min( m_A, n_A );
  buff_A   = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  ldim_A   = max( 1, m_A );

  buff_p   = ( int * ) malloc( n_A * sizeof( int ) );

  buff_tau = ( double * ) malloc( n_A * sizeof( double ) );

  buff_Q   = ( double * ) malloc( m_A * mn_A * sizeof( double ) );
  ldim_Q   = max( 1, m_A );

  // Generate binary file which stores the matrix (out of core)
  matrix_generate_ooc( m_A, n_A, A_fname ); 
  A_fp = fopen( A_fname, "r" );

  // transfer matrix to in-core TODO: remove this once building/debugging is complete
  read_check = fread( buff_A, sizeof( double ), m_A * n_A, A_fp );
  if ( read_check != m_A * n_A ) {
    printf( "Warning! file read failed \n" );
	return 1;
  }

  fseek( A_fp, 0, SEEK_SET );


#ifdef PRINT_DATA
  print_double_matrix( "ai", m_A, n_A, buff_A, ldim_A );
  print_double_vector( "taui", n_A, buff_tau );
#endif

  // Initialize vector with pivots.
  set_pvt_to_zero( n_A, buff_p );
  buff_p[ 0 ] = 0;
  buff_p[ 1 ] = 1;
  buff_p[ 2 ] = 1;
  buff_p[ 3 ] = 0;
  buff_p[ 4 ] = 1;
#ifdef PRINT_DATA
  print_int_vector( "pi", n_A, buff_p );
#endif

  // Create workspace.
  lwork       = max( 1, 128 * n_A );
  buff_wk_qp4 = ( double * ) malloc( lwork * sizeof( double ) );

  // Factorize matrix.
  printf( "%% Just before computing factorization.\n" );
  // New factorization.
  hqrrp_ooc( A_fname, m_A, n_A, buff_A, ldim_A, buff_p, buff_tau, 
           nb_alg, pp, 1 );
  // Current factorization.
  // dgeqp3_( & m_A, & n_A, buff_A, & ldim_A, buff_p, buff_tau, 
  //          buff_wk_qp4, & lwork, & info );
  printf( "%% Just after computing factorization.\n" );

  printf( "%% Info after factorization:      %d \n", info );
  printf( "%% Work[ 0 ] after factorization: %d \n", ( int ) buff_wk_qp4[ 0 ] );

  // Remove workspace.
  free( buff_wk_qp4 );

  // Build matrix Q.
  lwork     = max( 1, 128 * n_A );
  buff_wk_orgqr = ( double * ) malloc( lwork * sizeof( double ) );
  dlacpy_( "All", & m_A, & mn_A, buff_A, & ldim_A, buff_Q, & ldim_Q );
  dorgqr_( & m_A, & mn_A, & mn_A, buff_Q, & ldim_Q, buff_tau,
           buff_wk_orgqr, & lwork, & info );
  if( info != 0 ) {
    fprintf( stderr, "Error in dorgqr: Info: %d\n", info );
  }
  free( buff_wk_orgqr );

  // Print results.
#ifdef PRINT_DATA
  print_double_matrix( "af", m_A, n_A, buff_A, ldim_A );
  print_int_vector( "pf", n_A, buff_p );
  print_double_vector( "tauf", n_A, buff_tau );
  print_double_matrix( "qf", m_A, mn_A, buff_Q, ldim_Q );
#endif

  // remove file that stored matrix
  fclose( A_fp );
  remove( A_fname );

  // Free matrices and vectors.
  free( buff_A );
  free( buff_p );
  free( buff_tau );
  free( buff_Q );

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

// ============================================================================
static void matrix_generate( int m_A, int n_A, double * buff_A, int ldim_A ) {
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

// ============================================================================
static void print_double_vector( char * name, int n_v, double * buff_v ) {
  int  i, j;

  printf( "%s = [\n", name );
  for( i = 0; i < n_v; i++ ) {
    printf( "%le\n", buff_v[ i ] );
  }
  printf( "\n" );
  printf( "];\n" );
}

// ============================================================================
static void print_int_vector( char * name, int n_v, int * buff_v ) {
  int  i, j;

  printf( "%s = [\n", name );
  for( i = 0; i < n_v; i++ ) {
    printf( "%d\n", buff_v[ i ] );
  }
  printf( "];\n" );
}

// ============================================================================
static void init_pvt( int n_p, int * buff_p ) {
  int  i;

  for( i = 0; i < n_p; i++ ) {
    buff_p[ i ] = ( i + 1 );
  }
}

// ============================================================================
static void set_pvt_to_zero( int n_p, int * buff_p ) {
  int  i;

  for( i = 0; i < n_p; i++ ) {
    buff_p[ i ] = 0;
  }
}

