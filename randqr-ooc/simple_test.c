#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <time.h>
#include <mkl.h>
#include "hqrrp_ooc.h"
#include "NoFLA_HQRRP_WY_blk_var4.h"

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )

//#define PRINT_DATA
#define CHECK_OOC

// ============================================================================
// Declaration of local prototypes.

static void matrix_generate_ooc( int m_A, int n_A, char * dir_name, char * A_fname );

static void matrix_generate( int m_A, int n_A, double * buff_A, int ldim_A );

static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A );

static void print_double_vector( char * name, int n, double * vector );

static void print_int_vector( char * name, int n, int * vector );

static void init_pvt( int n, int * vector );

static void set_pvt_to_zero( int n_p, int * buff_p );


// ============================================================================
int main( int argc, char *argv[] ) {
  int     nb_alg, pp, k, m_A, n_A, mn_A, ldim_A, ldim_Q, info, lwork;
  double  * buff_A, * buff_tau, * buff_Q, * buff_wk_qp4, * buff_wk_orgqr;
  double  * buff_Ac, * buff_tauc;
  int     * buff_p;
  int     * buff_pc;
  FILE	  * A_fp; // pointer to the file that stores A
  char    dir_name[] = "./"; //"/media/hdd/";
  char    A_fname[] = "A_mat";

  char file_path[ sizeof( dir_name ) / sizeof( dir_name[0] ) + 
		sizeof( A_fname ) / sizeof( A_fname[0] ) ];
  strcpy( file_path, dir_name );
  strcat( file_path, A_fname );

  size_t  read_check;
  int     eq_check = 1;
  int i;

  struct timespec t1, t2;
  uint64_t diff;
  double t_ooc_fact = 0.0;

  // Create matrix A, vector p, vector s, and matrix Q.
  m_A      = 100;
  n_A      = 100;
  nb_alg   = 18;
  k        = n_A;
  pp	   = 0;

  mn_A     = min( m_A, n_A );
  buff_A   = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  ldim_A   = max( 1, m_A );

  buff_Ac  = ( double * ) malloc( m_A * n_A * sizeof( double ) );

  buff_p   = ( int * ) malloc( n_A * sizeof( int ) );

  buff_pc   = ( int * ) malloc( n_A * sizeof( int ) );

  buff_tau = ( double * ) malloc( n_A * sizeof( double ) );

  buff_tauc = ( double * ) malloc( n_A * sizeof( double ) );

  buff_Q   = ( double * ) malloc( m_A * mn_A * sizeof( double ) );
  ldim_Q   = max( 1, m_A );

  // Generate binary file which stores the matrix (out of core)
  matrix_generate_ooc( m_A, n_A, dir_name, A_fname ); 

  // transfer matrix to in-core
  A_fp = fopen( file_path, "r" );
  read_check = fread( buff_A, sizeof( double ), m_A * n_A, A_fp );
  fseek( A_fp, 0, SEEK_SET );
  read_check = fread( buff_Ac, sizeof( double ), m_A * n_A, A_fp );
  if ( read_check != m_A * n_A ) {
    printf( "Warning! file read failed \n" );
	return 1;
  }

  fclose( A_fp );


#ifdef PRINT_DATA
  print_double_matrix( "ai", m_A, n_A, buff_A, ldim_A );
  print_double_vector( "taui", n_A, buff_tau );
#endif

  // Initialize vector with pivots.
  set_pvt_to_zero( n_A, buff_p );
  for ( i=0; i < min(m_A,n_A); i++ ) {
    buff_p[ i ] = i;
	buff_pc[ i ] = i;
  }

#ifdef PRINT_DATA
  print_int_vector( "pi", n_A, buff_p );
#endif

  // Create workspace.
  lwork       = max( 1, 128 * n_A );
  buff_wk_qp4 = ( double * ) malloc( lwork * sizeof( double ) );

  // Factorize matrix.
  printf( "%% Just before computing factorization.\n" );
  // New factorization.

  clock_gettime( CLOCK_MONOTONIC, & t1 );

  hqrrp_ooc( dir_name, A_fname, m_A, n_A, ldim_A, buff_p, buff_tau, 
           nb_alg, k, pp, 1 );

  clock_gettime( CLOCK_MONOTONIC, & t2 );
  diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
  t_ooc_fact += ( double ) diff / (1E9);

  // Current factorization.
  NoFLA_HQRRP_WY_blk_var4( m_A, n_A, buff_Ac, ldim_A,
				buff_pc, buff_tauc, 
				nb_alg, pp, k, 1 );

  printf( "%% Just after computing factorization.\n" );
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

  // check whether OOC gets same results as in core
#ifdef CHECK_OOC
  A_fp = fopen( file_path, "r" );
  for ( i=0; i < n_A; i++ ) {
    fseek( A_fp, ( 0 +  i * ldim_A ) * sizeof( double ), SEEK_SET );
    read_check = fread( & buff_A[ 0 + i * ldim_A ], sizeof( double ), m_A, A_fp );
  }
  fclose( A_fp );
  
  for ( i=0; i < m_A * n_A; i++ ) {
    if ( abs( buff_A[ i ] - buff_Ac[ i ] ) > ( 1E-12 ) ) eq_check = 0; 
  }

  if ( eq_check == 1 ) {
    printf( "Success! in-core and out-of-core versions give the same result \n" );
  }
  else {
    printf( "Failure! in-core and out-of-core versions give different results! \n" );
  }

#endif

  // print out time required for ooc factorization
  printf( "Time required for hqrrp_ooc: %le\n", t_ooc_fact );


  // Print results.
#ifdef PRINT_DATA
  A_fp = fopen( file_path, "r" );
  for ( i=0; i < n_A; i++ ) {
    fseek( A_fp, ( 0 + i * ldim_A ) * sizeof( double ), SEEK_SET );
    fread( & buff_A[ 0 + i * ldim_A ], sizeof( double ), m_A, A_fp );
  }
  fclose( A_fp );

  print_double_matrix( "af", m_A, n_A, buff_A, ldim_A );
  print_double_matrix( "af2", m_A, n_A, buff_Ac, ldim_A );
  print_int_vector( "pf", n_A, buff_p );
  print_int_vector( "pf2", n_A, buff_pc );
  print_double_vector( "tauf", n_A, buff_tau );
  print_double_matrix( "qf", m_A, mn_A, buff_Q, ldim_Q );
#endif

  // remove file that stored matrix
  remove( file_path );

  // Free matrices and vectors.
  free( buff_A );
  free( buff_p );
  free( buff_tau );
  free( buff_Q );

  free( buff_Ac );
  free( buff_pc );
  free( buff_tauc );

  printf( "%% End of Program\n" );

  return 0;
}

// ============================================================================
static void matrix_generate_ooc( int m_A, int n_A, char * dir_name, char * A_fname ) {
  // populate the empty file pointed to by A_fp with a matrix
  // with random values 

  FILE * A_fp;
  double * col_p; // for storing one col at a time before transferring to disk
  int i,j;
  char file_path[ sizeof( dir_name ) / sizeof( dir_name[0] ) + 
		sizeof( A_fname ) / sizeof( A_fname[0] ) ];

  strcpy( file_path, dir_name );
  strcat( file_path, A_fname );

  A_fp = fopen( file_path, "w" );
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

