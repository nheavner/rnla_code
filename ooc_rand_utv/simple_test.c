#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "NoFLA_UTV_WY_blk_var2.h"
#include "rand_utv_ooc.h"

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) < (b) ? (a) : (b) )

//#define PRINT_DATA
#define CHECK_OOC

// ============================================================================
// Declaration of local prototypes.

static void matrix_generate( int m_A, int n_A, double * buff_A, int ldim_A );

static void matrix_generate_ooc( int m_A, int n_A, char file_path[] );

static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A );


// ============================================================================
int main( int argc, char *argv[] ) {
  int     bl_size, pp, n_iter, m_A, n_A, mn_A, ldim_A, ldim_U, ldim_V;
  int     i;
  double  * buff_A, * buff_Ac;
  double  * buff_V, * buff_Vc;
  double  * buff_U, * buff_Uc;

  FILE	  * A_fp; // pointer to the file that stores A
  FILE	  * V_fp; // pointer to the file that stores V
  FILE	  * U_fp; // pointer to the file that stores U
  char    dir_name[] = "./"; // "/media/hdd/"
  char    A_fname[]  = "A_mat";
  char    V_fname[]  = "V_out";
  char    U_fname[]  = "U_out";

  size_t read_check;
  int    eq_check_A = 1, eq_check_U = 1, eq_check_V = 1;
  int    V_check  = 1;
  int    U_check  = 1;

  char file_path[ sizeof( dir_name ) / sizeof( dir_name[0] ) + 
		sizeof( A_fname ) / sizeof( A_fname[0] ) ];
  strcpy( file_path, dir_name );
  strcat( file_path, A_fname );

  char file_path_V[ sizeof( dir_name ) / sizeof( dir_name[0] ) + 
		sizeof( V_fname ) / sizeof( V_fname[0] ) ];
  strcpy( file_path_V, dir_name );
  strcat( file_path_V, V_fname );

  char file_path_U[ sizeof( dir_name ) / sizeof( dir_name[0] ) + 
		sizeof( U_fname ) / sizeof( U_fname[0] ) ];
  strcpy( file_path_U, dir_name );
  strcat( file_path_U, U_fname );

  // allocate memory for matrix A.
  m_A      = 100;
  n_A      = 100;
  bl_size  = 12;
  pp       = 0;
  n_iter   = 1;
  mn_A     = min( m_A, n_A );

  buff_A   = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  buff_Ac  = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  ldim_A   = max( 1, m_A );

  buff_V   = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  buff_Vc  = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  ldim_V   = max( 1, n_A );

  buff_U   = ( double * ) malloc( m_A * m_A * sizeof( double ) );
  buff_Uc  = ( double * ) malloc( m_A * m_A * sizeof( double ) );
  ldim_U   = max( 1, m_A );

  // Generate binary file which stores matrix (out of core)
  matrix_generate_ooc( m_A, n_A, file_path );

  // transfer matrix to in-core
  A_fp = fopen( file_path, "r" );
  read_check = fread( buff_A, sizeof( double ), m_A * n_A, A_fp );
  if ( read_check != m_A * n_A ) {
    printf( "Warning! file read failed \n" );
	return 1;
  }
  fclose( A_fp );

#ifdef PRINT_DATA
  print_double_matrix( "ai", m_A, n_A, buff_A, ldim_A );
#endif

  // Factorize matrix.
  printf( "%% Just before computing factorization.\n" );

  // New factorization.
  rand_utv_ooc( dir_name, sizeof(dir_name), A_fname, sizeof(A_fname),
				m_A, n_A, ldim_A, 
				V_check, n_A, n_A, ldim_V, 
				U_check, m_A, m_A, ldim_U, 
				bl_size, pp, n_iter );
  printf( "%% Just after computing factorization.\n" );

  // old factorization
  NoFLA_UTV_WY_blk_var2( m_A, n_A, buff_A, ldim_A,
					 U_check, m_A, m_A, buff_U, m_A,
					 V_check, n_A, n_A, buff_V, n_A,
					 bl_size, pp, n_iter );

  printf( "%% Just after computing factorization.\n" );

#ifdef CHECK_OOC

  // check that in-core and out-of-core versions are the same
  A_fp = fopen( file_path, "r" );
  read_check = fread( buff_Ac, sizeof( double ), m_A * n_A, A_fp );
  if ( read_check != m_A * n_A ) {
    printf( "Warning! file read failed \n" );
	return 1;
  }
  fclose( A_fp );

  for ( i=0; i < m_A * n_A; i++ ) {
    if ( fabs( buff_A[ i ] - buff_Ac[ i ] ) > ( 1E-12 )  ||
		 isfinite( buff_Ac[ i ] ) == 0 ) 
	  eq_check_A = 0; 
  }

  if ( V_check == 1 ) {
    V_fp = fopen( file_path_V,"r" ); 
	read_check = fread( buff_Vc, sizeof( double ), n_A * n_A, V_fp );
	if ( read_check != n_A * n_A ) {
	  printf( "Warning! file read of V_out failed \n" );
	  return 1;
	}
	fclose( V_fp );

	for ( i=0; i < n_A * n_A; i++ ) {
	  if ( fabs( buff_V[ i ] - buff_Vc[ i ] ) > ( 1E-12 )  ||
		   isfinite( buff_Vc[ i ] ) == 0 ) 
		eq_check_V = 0; 
	}
  }

  if ( U_check == 1 ) {
    U_fp = fopen( file_path_U,"r" ); 
	read_check = fread( buff_Uc, sizeof( double ), m_A * m_A, U_fp );
	if ( read_check != m_A * m_A ) {
	  printf( "Warning! file read of U_out failed \n" );
	  return 1;
	}
	fclose( U_fp );

	for ( i=0; i < m_A * m_A; i++ ) {
	  if ( fabs( buff_U[ i ] - buff_Uc[ i ] ) > ( 1E-12 )  ||
		   isfinite( buff_Uc[ i ] ) == 0 ) { 
		eq_check_U = 0; 
	  }
	}
  }


  if ( eq_check_A == 1 ) {
    printf( "Success! in-core and out-of-core versions give the same result for A \n" );
  }
  else {
    printf( "Failure! in-core and out-of-core versions give different results for A! \n" );
  }

  if ( eq_check_V == 1 ) {
    printf( "Success! in-core and out-of-core versions give the same result for V \n" );
  }
  else {
    printf( "Failure! in-core and out-of-core versions give different results for V! \n" );
  }

  if ( eq_check_U == 1 ) {
    printf( "Success! in-core and out-of-core versions give the same result for U \n" );
  }
  else {
    printf( "Failure! in-core and out-of-core versions give different results for U! \n" );
  }

#endif

  // Free matrices and vectors.
  free( buff_A );
  free( buff_Ac );
  free( buff_V );
  free( buff_Vc );
  free( buff_U );
  free( buff_Uc );

  remove( file_path );

  printf( "%% End of Program\n" );

  return 0;
}

// ============================================================================
static void matrix_generate_ooc( int m_A, int n_A, char file_path[] ) {
  // populate the empty file pointed to by A_fp with a matrix
  // with random values 

  FILE * A_fp;
  double * col_p; // for storing one col at a time before transferring to disk
  int i,j;

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

