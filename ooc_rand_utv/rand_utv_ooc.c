/*
===============================================================================
Authors
===============================================================================

Per-Gunnar Martinsson
  Dept. of Applied Mathematics, 
  University of Colorado at Boulder, 
  526 UCB, Boulder, CO 80309-0526, USA

Gregorio Quintana-Orti
  Depto. de Ingenieria y Ciencia de Computadores, 
  Universitat Jaume I, 
  12.071 Castellon, Spain

Nathan Heavner
  Dept. of Applied Mathematics, 
  University of Colorado at Boulder, 
  526 UCB, Boulder, CO 80309-0526, USA

===============================================================================
Copyright
===============================================================================

Copyright (C) 2016, 
  Universitat Jaume I,
  University of Colorado at Boulder.

===============================================================================
Disclaimer
===============================================================================

This code is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY EXPRESSED OR IMPLIED.

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <mkl.h>
#include "rand_utv_ooc.h"

// ============================================================================
// Definition of macros.

#define max( a, b )  ( (a) > (b) ? (a) : (b) )
#define min( a, b )  ( (a) > (b) ? (b) : (a) )
#define dabs( a )    ( (a) >= 0.0 ? (a) : -(a) )


// ============================================================================
// Compilation declarations.

#define PROFILE
//#define PROFILE_FOR_GRAPH

// ============================================================================
// Declaration of local prototypes.

static int NoFLA_Set_to_identity( int m_A, int n_A, double * buff_A,
               int ldim_A );

static int NoFLA_Zero_strict_lower_triangular( int m_A, int n_A,
               double * buff_A, int ldim_A );

static int NoFLA_Zero( int m_A, int n_A, double * buff_A, int ldim_A );

static int NoFLA_Copy_vector_into_diagonal( double * v, int m_A, int n_A,
               double * buff_A, int ldim_A );

static int NoFLA_Multiply_BAB(
               char transa, char transb,
               int m_A, int n_A, double * buff_A, int ldim_A,
               int m_B, int n_B, double * buff_B, int ldim_B );

static int NoFLA_Multiply_BBA(
               char transa, char transb,
               int m_A, int n_A, double * buff_A, int ldim_A,
               int m_B, int n_B, double * buff_B, int ldim_B );

static int NoFLA_Compute_svd(
               int m_A, int n_A, double * buff_A, int ldim_A,
               int m_U, int n_U, double * buff_U, int ldim_U,
               int n_sv, double * buff_sv,
               int m_V, int n_V, double * buff_V, int ldim_V,
               int nb_alg );

static int NoFLA_Normal_random_matrix( int m_A, int n_A,
               double * buff_A, int ldim_A );

static double NoFLA_Normal_random_number( double mu, double sigma );

static int NoFLA_Apply_Q_WY_lhfc_blk_var2(
               int m_U, int n_U, double * buff_U, int ldim_U,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_B, int n_B, double * buff_B, int ldim_B );

static int NoFLA_Apply_Q_WY_rnfc_blk_var2(
               int m_U, int n_U, double * buff_U, int ldim_U,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_B, int n_B, double * buff_B, int ldim_B );

static int NoFLA_QRP_WY_unb_var2( int pivoting, int num_stages,
               int m_A, int n_A, double * buff_A, int ldim_A,
               double * buff_T, int ldim_T );

static int NoFLA_QRP_compute_norms(
               int m_A, int n_A, double * buff_A, int ldim_A,
               double * buff_d, double * buff_e );

static int NoFLA_QRP_downdate_partial_norms( int m_A, int n_A,
               double * buff_d,  int st_d,
               double * buff_e,  int st_e,
               double * buff_wt, int st_wt,
               double * buff_A,  int ldim_A );

static int NoFLA_QRP_pivot_G( int j_max_col,
               int m_G, double * buff_G, int ldim_G,
               double * buff_d, double * buff_e );

static int Mult_ABt_A_out( int m, int n, int k,
				FILE * A_fp, int ldim_A, 
				double * B_p, int ldim_B,
				double * C_p, int ldim_C,
				int bl_size, int num_cols_read,
				double * t_read, double * t_write, double * t_init_Y );

static int Mult_BtA_A_out( int m, int n, int k,
				FILE * A_fp, int ldim_A, 
				double * B_p, int ldim_B,
				double * C_p, int ldim_C,
				int bl_size, int num_cols_read,
				double * t_read, double * t_write, double * t_init_Y );

static int Set_to_identity_ooc( int m_A, int n_A, FILE * A_fp, int ldim_A,
								int num_cols_read );

static struct timespec start_timer( void ); 

static double stop_timer( struct timespec t1 ); 

static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A ) {
  int  i, j;

  printf( "%s = [\n", name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%le ", buff_A[ i + j * ldim_A ] );
    }
    printf( ";\n" );
  }
  printf( "];\n" );
}

// ============================================================================
int rand_utv_ooc(
		char dir_name[], size_t dir_name_size, char A_fname[], size_t A_fname_size,
        int m_A, int n_A, int ldim_A,
		int build_v, int m_V, int n_V, int ldim_V,
		int build_u, int m_U, int n_U, int ldim_U,
        int nb_alg, int pp, int n_iter ) {
//
// randUTV_ooc: It computes the UTV factorization of matrix A when A is stored out of core.
//
// Main features:
//   * BLAS-3 based.
//   * Compact WY transformations are used instead of UT transformations.
//   * No use of libflame.
//
// Matrices A, U, and V must be stored in column-order.
//
// Arguments:
// ----------
// m_A:      Number of rows of matrix A.
// n_A:      Number of columns of matrix A.
// buff_A:   Address of data in matrix A. Matrix to be factorized.
// ldim_A:   Leading dimension of matrix A.
// build_u:  If build_u==1, matrix U is built.
// m_U:      Number of rows of matrix U.
// n_U:      Number of columns of matrix U.
// buff_U:   Address of data in matrix U.
// ldim_U:   Leading dimension of matrix U.
// build_v:  If build_v==1, matrix V is built.
// m_V:      Number of rows of matrix V.
// n_V:      Number of columns of matrix V.
// buff_V:   Address of data in matrix V.
// ldim_V:   Leading dimension of matrix V.
// nb_alg:   Block size. Usual values for nb_alg are 32, 64, etc.
// pp:       Oversampling size. Usual values for pp are 5, 10, etc.
// n_iter:   Number of "power" iterations. Usual values are 2.
//
// Final comments:
// ---------------
// This code has been created from a libflame code. Hence, you can find some
// commented calls to libflame routines. We have left them to make it easier
// to interpret the meaning of the C code.
//
  // Declaration of variables.
  double  d_one = 1.0, d_zero = 0.0;
  char    all = 'A', t = 'T', n = 'N';
  double  * buff_G, * buff_Yt, * buff_Y, * buff_S1, * buff_S2,
          * buff_SU, * buff_sv, * buff_SVT,
          * buff_SUtl, * buff_svl, * buff_SVTtl,
		  * buff_A_mid, // A_mid = [A12; A22; A32]
		  * buff_A23,
		  * buff_work_row, * buff_work_col, // for storing chunks of A as it's read out to apply transforms
          * buff_GBl, * buff_YtBl, * buff_YBl, * buff_S1tl, * buff_S2tl,
          * buff_C1, * buff_D1, * buff_CR, * buff_DR;
  int     i, j, k, bRow, mn_A;
  int     ldim_Yt, ldim_Y, ldim_G, ldim_S1, ldim_S2, ldim_SU, ldim_SVT, ldim_SA,
		  ldim_A_mid, ldim_work_row, ldim_work_col,
		  ldim_A23;
  int     m_YtBl, n_YtBl, m_YBl, n_YBl, m_GBl, n_GBl, m_CR, n_CR,
          m_WR, n_WR, m_XR, n_XR,
		  m_A_midl, n_A_midl, m_work_rowl, n_work_rowl, m_work_coll, n_work_coll,
		  m_A23l, n_A23l;

  FILE * A_fp; // pointer to the file that stores A
  char file_path[ dir_name_size / sizeof( dir_name[0] ) + 
				  A_fname_size / sizeof( A_fname[0] ) ];
  strcpy( file_path, dir_name );
  strcat( file_path, A_fname );

  FILE * V_fp;
  char V_out_name[] = "V_out";
  char V_out_path[ dir_name_size / sizeof( dir_name[0] ) + 
				   sizeof( V_out_name ) / sizeof( V_out_name[0] ) ];
  strcpy( V_out_path, dir_name );
  strcat( V_out_path, V_out_name );

  FILE * U_fp;
  char U_out_name[] = "U_out";
  char U_out_path[ dir_name_size / sizeof( dir_name[0] ) + 
				   sizeof( U_out_name ) / sizeof( U_out_name[0] ) ];
  strcpy( U_out_path, dir_name );
  strcat( U_out_path, U_out_name );

  int num_cols_read = nb_alg * 2; // TODO: make this value better
  int num_cols_readl; // for local use in loops

  int num_rows_read = nb_alg * 2; // TODO: make this value better
  int num_rows_readl;

  int err_check;

#ifdef PROFILE
  struct timespec t1;
  double  tt_by,
          tt_qr1_fact, tt_qr1_updt_a, tt_qr1_updt_v,
          tt_qr2_fact, tt_qr2_updt_a, tt_qr2_updt_u,
          tt_svd_fact, tt_svd_updt_a, tt_svd_updt_uv;
  double  tt_read, tt_write;
#endif

  // Executable Statements.

  // Set seed for random generator.
  srand( 12 );

#ifdef PROFILE
  tt_by          = 0.0;
  tt_qr1_fact    = 0.0;
  tt_qr1_updt_a  = 0.0;
  tt_qr1_updt_v  = 0.0;
  tt_qr2_fact    = 0.0;
  tt_qr2_updt_a  = 0.0;
  tt_qr2_updt_u  = 0.0;
  tt_svd_fact    = 0.0;
  tt_svd_updt_a  = 0.0;
  tt_svd_updt_uv = 0.0;
  tt_read        = 0.0;
  tt_write       = 0.0;
#endif

  // Create and initialize auxiliary objects.
  A_fp = fopen( file_path, "r+" );
  V_fp = fopen( V_out_path, "w+" );
  U_fp = fopen( U_out_path, "w+" );

  buff_G   = ( double * ) malloc( m_A * ( nb_alg + pp ) * sizeof( double ) );
  buff_Yt  = ( double * ) malloc( n_A * ( nb_alg + pp ) * sizeof( double ) );
  buff_Y   = ( double * ) malloc( n_A * ( nb_alg + pp ) * sizeof( double ) );
  buff_S1  = ( double * ) malloc( nb_alg * nb_alg * sizeof( double ) );
  buff_S2  = ( double * ) malloc( nb_alg * nb_alg * sizeof( double ) );
  buff_SU  = ( double * ) malloc( nb_alg * nb_alg * sizeof( double ) );
  buff_sv  = ( double * ) malloc( nb_alg * sizeof( double ) );
  buff_SVT = ( double * ) malloc( nb_alg * nb_alg * sizeof( double ) );
  buff_A_mid = ( double * ) malloc( m_A * nb_alg * sizeof( double ) );
  buff_A23 = ( double * ) malloc( nb_alg * n_A * sizeof( double ) );
  buff_work_row = ( double * ) malloc( n_A * num_rows_read * sizeof( double ) );
  buff_work_col = ( double * ) malloc( m_A * num_cols_read * sizeof( double ) );

  // Some initializations.
  ldim_G     = m_A;
  ldim_Yt    = nb_alg + pp;
  ldim_Y     = n_A;
  ldim_S1    = nb_alg;
  ldim_S2    = nb_alg;
  ldim_SU    = nb_alg;
  ldim_SVT   = nb_alg;
  ldim_SA    = nb_alg;
  ldim_A_mid = m_A;
  ldim_A23 = nb_alg;
  ldim_work_row  = num_rows_read;
  ldim_work_col  = m_A;
  buff_SUtl  = & buff_SU [ 0 + 0 * ldim_SU ];
  buff_svl   = & buff_sv[ 0 ];
  buff_SVTtl = & buff_SVT[ 0 + 0 * ldim_SVT ];

  // %%% Initialize U and V 
  // U = eye(m);
  // V = eye(n);
  if ( build_v == 1 ) {
    Set_to_identity_ooc( n_A, n_A, V_fp, n_A, num_cols_read );
  }

  if ( build_u == 1 ) {
    Set_to_identity_ooc( m_A, m_A, U_fp, m_A, num_cols_read );
  }


  // Main Loop.
  mn_A = min( m_A, n_A );
  for ( i = 0; i < mn_A; i += nb_alg ) {
    bRow = min( nb_alg, mn_A - i );

    // Some initializations for every iteration.
    m_YtBl = bRow + pp; 
    n_YtBl = n_A - i;
    m_YBl = n_A - i;
    n_YBl = bRow + pp;
    m_GBl = m_A - i;
    n_GBl = bRow + pp;
    m_CR  = m_A;
    n_CR  = n_A - i;
    m_WR  = m_A;
    n_WR  = m_A - i;
    m_XR  = n_A;
    n_XR  = n_A - i;

	m_A_midl = m_A - i;
	n_A_midl = bRow;
	m_A23l   = bRow;
	n_A23l   = n_A - i - bRow;
	m_work_rowl  = nb_alg;
    n_work_rowl  = n_A - i;
	m_work_coll  = m_A - i;
    n_work_coll  = nb_alg;

    buff_GBl  = & buff_G[ i + 0 * ldim_G ];
    buff_YtBl  = & buff_Yt[ 0 + i * ldim_Yt ];
    buff_YBl  = & buff_Y[ i + 0 * ldim_Y ];
    buff_S1tl = & buff_S1[ 0 + 0 * ldim_S1 ];
    buff_S2tl = & buff_S2[ 0 + 0 * ldim_S2 ];

    // %%% Compute the "sampling" matrix Y.
    // Aloc = T([J2,I3],[J2,J3]);
    // Y    = Aloc'*randn(m-(j-1)*b,b+p);
#ifdef PROFILE
    t1 = start_timer();
#endif
    NoFLA_Normal_random_matrix( m_GBl, n_GBl, buff_GBl, ldim_G );

	fseek( A_fp, ( 0 + ( i ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
    Mult_BtA_A_out( m_YtBl, n_YtBl, m_GBl,
				A_fp, ldim_A, 
				buff_GBl, ldim_G,
				buff_YtBl, ldim_Yt,
				nb_alg, num_cols_read,
				& tt_read, & tt_write, & tt_by );

    // %%% Perform "power iteration" if requested.
    // for i_iter = 1:n_iter
    //   Y = Aloc'*(Aloc*Y);
    // end
    for( j = 0; j < n_iter; j++ ) {
      // Reuse GBl.
      // FLA_Gemm( FLA_NO_TRANSPOSE, FLA_NO_TRANSPOSE,
      //           FLA_ONE, ABR, YtBl, FLA_ZERO, GBl );
	  fseek( A_fp, ( 0 + ( i ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
      Mult_ABt_A_out( m_GBl, n_GBl, n_YtBl,
				A_fp, ldim_A, 
				buff_YtBl, ldim_Yt,
				buff_GBl, ldim_G,
				nb_alg, num_cols_read,
				& tt_read, & tt_write, & tt_by );

	  fseek( A_fp, ( 0 + ( i ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
	  Mult_BtA_A_out( m_YtBl, n_YtBl, m_GBl,
				  A_fp, ldim_A, 
				  buff_GBl, ldim_G,
				  buff_YtBl, ldim_Yt,
				  nb_alg, num_cols_read,
				  & tt_read, & tt_write, & tt_by );
    }
	
#ifdef PROFILE
    tt_by += stop_timer(t1);
	t1 = start_timer();
#endif

    // %%% Construct the local transform to be applied on the right.
    // if (p > 0)
    //   [~,~,Jtmp] = qr(Y,0);
    //   [Vloc,~,~] = qr(Y(:,Jtmp(1:b)));
    // else
    //   [Vloc,~]   = LOCAL_nonpiv_QR(Y,b);
    // end
    // FLA_Part_2x2( S1,  & S1tl,  & None1,
    //                    & None2, & None3,   bRow, bRow, FLA_TL );
	//

	// copy transpose of Yt into Y
	// TODO: probably figure out a better way to do this
    for ( j=0; j < m_YtBl; j++ ) {
	  for ( k=0; k < n_YtBl; k++ ) {
	    buff_YBl[ k + j * ldim_Y ] = buff_YtBl[ j + k * ldim_Yt ];
	  }
	}
	
	// construct the local transform V to be applied on the right
    if( pp > 0 ) {
      NoFLA_QRP_WY_unb_var2( 1, bRow, m_YBl, n_YBl, buff_YBl, ldim_Y,
          buff_S1tl, ldim_S1 );
    } else {
      NoFLA_QRP_WY_unb_var2( 0, bRow, m_YBl, n_YBl, buff_YBl, ldim_Y,
          buff_S1tl, ldim_S1 );
    }

#ifdef PROFILE
    tt_qr1_fact += stop_timer(t1);
#endif

    // %%% Apply the pivot matrix to rotate maximal mass into the "J2" column.
    // T(:,[J2,J3])  = T(:,[J2,J3])*Vloc;
    // Update matrix A with transformations from the first QR.
	for ( j=0; j < m_A; j += num_rows_read ) {

      // read out a block of the matrix
	  num_rows_readl = min( num_rows_read, m_A - j ); 

	  for ( k=0; k < n_A - i; k += 1 ) {
	    fseek( A_fp, ( ( j ) + ( i + k ) * ldim_A ) * sizeof( double ), SEEK_SET );
		err_check = fread( & buff_work_row[ 0 + ( k ) * ldim_work_row ], sizeof( double ), 
						   num_rows_readl, A_fp );			
		if ( err_check != num_rows_readl ) {
		  printf( "Error! read of block for application of V failed!\n" );
		  return 1;
		}
	  }

	  for ( k=0; k < num_rows_readl; k += nb_alg ) {
	    // apply the transformations
		NoFLA_Apply_Q_WY_rnfc_blk_var2(
			m_YBl, n_YBl, buff_YBl, ldim_Y,
			bRow, bRow, buff_S1tl, ldim_S1,
			min( nb_alg, num_rows_readl - k ), n_A - i, 
			& buff_work_row[ k + 0 * (ldim_work_row) ], ldim_work_row );
      }

	  // write results back out
	  for ( k=0; k < n_A - i; k += 1 ) {
	    fseek( A_fp, ( ( j ) + ( i + k ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
		err_check = fwrite( & buff_work_row[ 0 + ( k ) * ldim_work_row ], sizeof( double ), 
						   num_rows_readl, A_fp );			
		if ( err_check != num_rows_readl ) {
		  printf( "Error! write of block for application of V failed!\n" );
		  return 1;
		}
	  }

	}

    // Update matrix V with transformations from the first QR.

    if( build_v == 1 ) {
	  
	  for ( j=0; j < m_V; j += num_rows_read ) {

		num_rows_readl = min( num_rows_read, m_V - j ); 

		// read out a block of V
		for ( k=0; k < n_V - i; k += 1 ) {
		  fseek( V_fp, ( ( j ) + ( i + k ) * ldim_V ) * sizeof( double ), SEEK_SET );
		  err_check = fread( & buff_work_row[ 0 + ( k ) * ldim_work_row ], sizeof( double ), 
							 num_rows_readl, V_fp );			
		  if ( err_check != num_rows_readl ) {
			printf( "Error! read of block for application of V failed!\n" );
			return 1;
		  }
		}

		for ( k=0; k < num_rows_readl; k += nb_alg ) {
		  // apply update one block at a time
		  NoFLA_Apply_Q_WY_rnfc_blk_var2(
			  m_YBl, bRow, buff_YBl,  ldim_Y,
			  bRow,  bRow, buff_S1tl, ldim_S1,
			  min( nb_alg, num_rows_readl - k ), n_V - i, 
			  & buff_work_row[ k + ( 0 ) * ldim_work_row ], ldim_work_row );
		}

	    // write out results
		for ( k=0; k < n_V - i; k += 1 ) {
		  fseek( V_fp, ( ( j ) + ( i + k ) * ldim_V ) * sizeof( double ), SEEK_SET );
		  err_check = fwrite( & buff_work_row[ 0 + ( k ) * ldim_work_row ], sizeof( double ), 
							 num_rows_readl, V_fp );			
		  if ( err_check != num_rows_readl ) {
			printf( "Error! write of block for application of V failed!\n" );
			return 1;
		  }
		}

      } 

	}
	

#ifdef PROFILE
    tt_qr1_updt_a += stop_timer(t1);
	t1 = start_timer();
#endif

    // %%% Next determine the rotations to be applied "from the left".
    // [Uloc,Dloc]      = LOCAL_nonpiv_QR(T([J2,I3],J2));

	// read out block [A12; A22; A32] and then factorize it
	fseek( A_fp, ( ( 0 ) + ( i ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
	err_check = fread( buff_A_mid, sizeof( double ), m_A * bRow, A_fp );			
	if ( err_check != m_A * bRow ) {
	  printf( "Error! read of A_mid failed!\n" );
	  return 1;
	}
    NoFLA_QRP_WY_unb_var2( 0, bRow, m_A_midl, n_A_midl, 
						   & buff_A_mid[i + 0 * ldim_A_mid], ldim_A_mid,
						   buff_S2tl, ldim_S2 );

#ifdef PROFILE
    tt_qr2_fact += stop_timer( t1 );
	t1 = start_timer();
#endif

    // Update rest of matrix A with transformations from the second QR.
    // MyFLA_Apply_Q_UT_lhfc_blk( A11, A21, S2tl, A12, A22 );
	for ( j=i+nb_alg; j < n_A; j += num_cols_read ) {
	  num_cols_readl = min( num_cols_read, n_A - j );

	  fseek( A_fp, ( ( 0 ) + ( j ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
	  err_check = fread( buff_work_col, sizeof( double ), m_A * num_cols_readl, A_fp );			

      for ( k=0; k < num_cols_readl; k += nb_alg ) {
		NoFLA_Apply_Q_WY_lhfc_blk_var2(
			m_A_midl, n_A_midl, & buff_A_mid[ i + ( 0 ) * ldim_A_mid ], ldim_A_mid,
			bRow, bRow, buff_S2tl, ldim_S2,
			m_work_coll, min( nb_alg, num_cols_readl - k ), 
			& buff_work_col[ i + ( k ) * ldim_work_col ], ldim_work_col );
      }

	  // write results back out
	  fseek( A_fp, ( ( 0 ) + ( j ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
	  err_check = fwrite( buff_work_col, sizeof( double ), 
						  m_A * num_cols_readl, A_fp );			
	}
	
	// update U with transformations from the second QR
    if( build_u == 1 ) {
	  for ( j=0; j < m_U; j += num_rows_read ) {

		num_rows_readl = min( num_rows_read, m_U - j ); 

		// read out a block of U
		for ( k=0; k < n_U - i; k += 1 ) {
		  fseek( U_fp, ( ( j ) + ( i + k ) * ldim_U ) * sizeof( double ), SEEK_SET );
		  err_check = fread( & buff_work_row[ 0 + ( k ) * ldim_work_row ], sizeof( double ), 
							 num_rows_readl, U_fp );			
		  if ( err_check != num_rows_readl ) {
			printf( "Error! read of block for application of U failed!\n" );
			return 1;
		  }
		}

		for ( k=0; k < num_rows_readl; k += nb_alg ) {
		  // apply update one block at a time
		  NoFLA_Apply_Q_WY_rnfc_blk_var2(
			  m_A_midl, n_A_midl, & buff_A_mid[ i + ( 0 ) * ldim_A_mid ], ldim_A_mid,
			  bRow, bRow, buff_S2tl, ldim_S2,
			  min( nb_alg, num_rows_readl - k ), n_U - i, 
			  & buff_work_row[ k + ( 0 ) * ldim_work_row ], ldim_work_row );
		}

	    // write out results
		for ( k=0; k < n_U - i; k += 1 ) {
		  fseek( U_fp, ( ( j ) + ( i + k ) * ldim_U ) * sizeof( double ), SEEK_SET );
		  err_check = fwrite( & buff_work_row[ 0 + ( k ) * ldim_work_row ], sizeof( double ), 
							 num_rows_readl, U_fp );			
		  if ( err_check != num_rows_readl ) {
			printf( "Error! write of block for application of U failed!\n" );
			return 1;
		  }
		}
      } // end j loop 

	} // end ( if build_u == 1 )

#ifdef PROFILE
    tt_qr2_updt_a += stop_timer( t1 );
	t1 = start_timer();
#endif

    // Compute miniSVD.
    // [Utmp,Dtmp,Wloc] = svd(Dloc(1:b,:));
    // Dloc(1:b,:)      = Dtmp;
    // Uloc(:,1:b)      = Uloc(:,1:b)*Utmp;
    // Vloc(:,1:b)      = Vloc(:,1:b)*Wloc; % Update Vloc.
    // T([J2,I3],J2)    = Dloc;
    // T(J1,J2)         = T(J1,J2)*Wloc;
    // T([J2,I3],J3)    = Uloc'*T([J2,I3],J3);
    //
    // %%% Store away the ON matrices.
    // U(:,[J2,I3]) = U(:,[J2,I3])*Uloc;
    // V(:,[J2,J3]) = V(:,[J2,J3])*Vloc;
    NoFLA_Zero_strict_lower_triangular( m_A_midl, n_A_midl, 
										& buff_A_mid[ i + 0 * ldim_A_mid ], ldim_A_mid );
    NoFLA_Compute_svd(
        bRow, bRow, & buff_A_mid[ i + 0 * ldim_A_mid ], ldim_A_mid,
        bRow, bRow, buff_SUtl, ldim_SU,
        bRow, buff_svl,
        bRow, bRow, buff_SVTtl, ldim_SVT,
        bRow );
    NoFLA_Zero( bRow, bRow, & buff_A_mid[ i + 0 * ldim_A_mid ], ldim_A_mid );
    NoFLA_Copy_vector_into_diagonal( buff_sv, bRow, bRow,	
									 & buff_A_mid[ i + 0 * ldim_A_mid ], ldim_A_mid );
#ifdef PROFILE
    tt_svd_fact += stop_timer( t1 );
	t1 = start_timer();
#endif

    // Apply U of miniSVD to A.

      // read out block A23
	  for ( j=0; j < n_A - i - bRow; j++ ) {
		fseek( A_fp, ( ( i ) + ( i + bRow + j ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
		err_check = fread( & buff_A23[ 0 + j * ldim_A23 ], sizeof( double ), bRow, A_fp );	
	  }

      // apply the transformation
	  NoFLA_Multiply_BAB( 't', 'n',
		  bRow, bRow, buff_SUtl, ldim_SU,
		  bRow, n_A23l, buff_A23,  ldim_A23 );

	  // write out the results
	  for ( j=0; j < n_A - i - bRow; j++ ) {
		fseek( A_fp, ( ( i ) + ( i + bRow + j ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
		err_check = fwrite( & buff_A23[ 0 + j * ldim_A23 ], sizeof( double ), 
						  bRow, A_fp );	
      }

    // Apply V of miniSVD to A.
    NoFLA_Multiply_BBA( 't', 'n',
        bRow, bRow, buff_SVTtl, ldim_SVT,
        i,    bRow, buff_A_mid,   ldim_A_mid );

    // write out A_mid
	fseek( A_fp, ( ( 0 ) + ( i ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
	err_check = fwrite( buff_A_mid, sizeof( double ), m_A * bRow, A_fp );			

#ifdef PROFILE
    tt_svd_updt_a += stop_timer( t1 );
#endif

    // update V
	if ( build_v == 1 ) {

	  // read out middle block of V
	  fseek( V_fp, ( 0 + ( i ) * ldim_V ) * sizeof( double ), SEEK_SET );
	  err_check = fread( buff_A_mid, sizeof( double ), m_V * bRow, V_fp );
				  // reuse A_mid to save memory; this works as long as m_A >= n_A
	  if ( err_check != m_V * bRow ) {
		printf( "Error! read of V_mid failed!\n" );
		return 1;
	  }

	  // apply V of miniSVD to global V
	  NoFLA_Multiply_BBA( 't', 'n',
		  bRow, bRow, buff_SVTtl, ldim_SVT,
		  m_V,  bRow, buff_A_mid, m_V );

	  // write out results
	  fseek( V_fp, ( 0 + ( i ) * ldim_V ) * sizeof( double ), SEEK_SET );
	  err_check = fwrite( buff_A_mid, sizeof( double ), m_V * bRow, V_fp );			
	}

	// Update U 
	if ( build_u == 1 ) {

	  // read out middle block of U
	  fseek( U_fp, ( 0 + ( i ) * ldim_U ) * sizeof( double ), SEEK_SET );
	  err_check = fread( buff_A_mid, sizeof( double ), m_U * bRow, U_fp );
				  // reuse A_mid to save memory; this works as long as m_A >= n_A
	  if ( err_check != m_U * bRow ) {
		printf( "Error! read of V_mid failed!\n" );
		return 1;
	  }

	  // apply U of miniSVD to global U
	  NoFLA_Multiply_BBA( 'n', 'n',
		  bRow, bRow, buff_SUtl, ldim_SU,
		  m_U,  bRow, buff_A_mid, m_U );

	  // write out results
	  fseek( U_fp, ( 0 + ( i ) * ldim_U ) * sizeof( double ), SEEK_SET );
	  err_check = fwrite( buff_A_mid, sizeof( double ), m_U * bRow, U_fp );			
	}
	
    
  } // End of main loop.

  // The last block is processed inside the previous loop.

  // Remove auxiliary objects.
  fclose( A_fp );
  fclose( V_fp );
  fclose( U_fp );

  free( buff_Yt );
  free( buff_Y );
  free( buff_G );
  free( buff_S1 );
  free( buff_S2 );
  free( buff_SU );
  free( buff_sv );
  free( buff_SVT );
  free( buff_A_mid );
  free( buff_A23 );
  free( buff_work_row );
  free( buff_work_col );

#ifdef PROFILE
  #ifdef PROFILE_FOR_GRAPH
    printf("%le %le %le %le %le %le %le \n", tt_by, 
								 tt_qr1_fact, 
								 tt_qr1_updt_a,
								 tt_qr2_fact, 
								 tt_qr2_updt_a,
								 tt_svd_fact,
								 tt_by +
								 tt_qr1_fact + tt_qr1_updt_a + tt_qr1_updt_v +
								 tt_qr2_fact + tt_qr2_updt_a + tt_qr2_updt_u +
								 tt_svd_fact + tt_svd_updt_a + tt_svd_updt_uv );
  #else
	printf( "%% tt_build_y:     %le\n", tt_by );
	printf( "%% tt_qr1:         %le\n", tt_qr1_fact + tt_qr1_updt_a +
										tt_qr1_updt_v );
	printf( "%%     tt_qr1_fact:    %le\n", tt_qr1_fact );
	printf( "%%     tt_qr1_updt_a:  %le\n", tt_qr1_updt_a );
	printf( "%%     tt_qr1_updt_v:  %le\n", tt_qr1_updt_v );
	printf( "%% tt_qr2:         %le\n", tt_qr2_fact + tt_qr2_updt_a +
										tt_qr2_updt_u );
	printf( "%%     tt_qr2_fact:    %le\n", tt_qr2_fact );
	printf( "%%     tt_qr2_updt_a:  %le\n", tt_qr2_updt_a );
	printf( "%%     tt_qr2_updt_u:  %le\n", tt_qr2_updt_u );
	printf( "%% tt_svd:         %le\n", tt_svd_fact + tt_svd_updt_a +
										tt_svd_updt_uv);
	printf( "%%     tt_svd_fact:    %le\n", tt_svd_fact );
	printf( "%%     tt_svd_updt_a:  %le\n", tt_svd_updt_a );
	printf( "%%     tt_svd_updt_uv: %le\n", tt_svd_updt_uv );
	printf( "%% total_time:     %le\n",
			tt_by +
			tt_qr1_fact + tt_qr1_updt_a + tt_qr1_updt_v +
			tt_qr2_fact + tt_qr2_updt_a + tt_qr2_updt_u +
			tt_svd_fact + tt_svd_updt_a + tt_svd_updt_uv );
  #endif
#endif

  return 0;
}

// ============================================================================
static int Mult_ABt_A_out( int m, int n, int k,
				FILE * A_fp, int ldim_A, 
				double * B_p, int ldim_B,
				double * C_p, int ldim_C,
				int bl_size, int num_cols_read,
				double * t_read, double * t_write, double * t_init_Y ) {
  // TODO: for this and Mult_ABt_A_out, clarify which portion of A is used in the multiplication
  //       (the bottom rows); i.e. this does NOT work in general for multiplying a general submatrix
  // for matrix A stored out of core and
  // matrices B and C stored in core, computes
  // C <-- A*Bt 

  // num_cols_read is the number of cols of A that can be stored in RAM at a time
  // A_fp must be pointed at the correct spot in the file on input

  // declare aux vars
  double * A_bl_p; // stores a block of cols of A
  int num_cols_readl, b;
  size_t check;

  double d_one = 1.0, d_zero = 0.0;
  int i,j;

#ifdef PROFILE
  struct timespec t1, t2;
  uint64_t diff;
#endif

  // some initializations
  A_bl_p = ( double * ) malloc( ldim_A * num_cols_read * sizeof( double ) );

  // C needs to be initialized to zeroes
  for ( j=0; j < n; j++ ) {
    for ( i=0; i < m; i++ ) {
	  C_p[ i + j * ldim_C ] = 0.0;
	}

  }

  // do multiplication one block at a time
  for ( i=0; i < k; i+= num_cols_read ) {
    
	num_cols_readl = min( num_cols_read, k - i );
	
	// read a block of A into memory

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t1 );
#endif
	  
	check = fread( A_bl_p, sizeof( double ), ldim_A * num_cols_readl, A_fp ); 
	if ( ( int ) check != ldim_A * num_cols_readl ) {
	  printf( "Warning! read failed in Mult_BA_A_out. check = %d \n", (int)check );  
	  return 1;
	}

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t2 );
  diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
  * t_read += ( double ) diff / (1E9);
#endif

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t1 );
#endif

    for ( j=0; j < num_cols_readl; j+= bl_size ) {

      b = min( bl_size, num_cols_readl - j );

	  // do multiplication; gives one block of cols of C
	  dgemm( "No transpose", "Transpose", 
			  & m, & n, & b,
			  & d_one, & A_bl_p[ ( ldim_A - k ) + ( j ) * ldim_A ], & ldim_A, 
			  & B_p[ ( 0 ) + ( i + j ) * ldim_B ], & ldim_B,
			  & d_one, C_p, & ldim_C );
    }

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t2 );
  diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
  * t_init_Y += ( double ) diff / (1E9);
#endif

  }

  // free memory
  free( A_bl_p );

}

// ============================================================================
static int Mult_BtA_A_out( int m, int n, int k,
				FILE * A_fp, int ldim_A, 
				double * B_p, int ldim_B,
				double * C_p, int ldim_C,
				int bl_size, int num_cols_read,
				double * t_read, double * t_write, double * t_init_Y ) {
  // for matrix A stored out of core and
  // matrices B and C stored in core, computes
  // C <-- B'*A if trans == 't' || 'T' (note this is the transpose of A'*B)

  // num_cols_read is the number of cols of A that can be stored in RAM at a time
  // A_fp must be pointed at the correct spot in the file on input

  // declare aux vars
  double * A_bl_p; // stores a block of cols of A
  int num_cols_readl, b;
  size_t check;

  double d_one = 1.0, d_zero = 0.0;
  int i,j;

#ifdef PROFILE
  struct timespec t1, t2;
  uint64_t diff;
#endif

  // some initializations
  A_bl_p = ( double * ) malloc( ldim_A * num_cols_read * sizeof( double ) );

  // do multiplication one block at a time
  for ( i=0; i < n; i+= num_cols_read ) {
    
	num_cols_readl = min( num_cols_read, n - i );

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t1 );
#endif
	  
	// read a block of A into memory
	check = fread( A_bl_p, sizeof( double ), ldim_A * num_cols_readl, A_fp ); 
	if ( ( int ) check != ldim_A * num_cols_readl ) {
	  printf( "Warning! read failed in Mult_BtA_A_out. check = %d \n", (int)check );  
	  return 1;
	}

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t2 );
  diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
  * t_read += ( double ) diff / (1E9);
#endif

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t1 );
#endif

    for ( j=0; j < num_cols_readl; j+= bl_size ) {

      b = min( bl_size, num_cols_readl - j );

	  // do multiplication; gives one block of cols of C
	  dgemm( "Transpose", "No transpose", 
			  & m, & b, & k,
			  & d_one, B_p, & ldim_B, 
			  & A_bl_p[ ( ldim_A - k ) + ( j ) * ldim_A  ], & ldim_A,
			  & d_zero, & C_p[ 0 + ( i + j ) * ldim_C ], & ldim_C );
    }

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t2 );
  diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
  * t_init_Y += ( double ) diff / (1E9);
#endif

  }

  // free memory
  free( A_bl_p );

}
// ============================================================================
static int Set_to_identity_ooc( int m_A, int n_A, FILE * A_fp, int ldim_A,
								int num_cols_read ) {
  // Set contents of file pointed to by A_fp to the identity matrix
  // this function assume m_A >= n_A

  int i, j;
  int num_cols_readl;
  int err_check;

  double * buff_cols = ( double * ) malloc( ldim_A * num_cols_read * sizeof( double ) );

  // initialize buff_zeroes matrix
  for ( i=0; i < ldim_A * num_cols_read; i++ ) {
    buff_cols[ i ] = 0.0;  
  }

  // write to A_fp one block at a time
  fseek( A_fp, 0, SEEK_SET );
  for ( j=0; j < n_A; j += num_cols_read ) {
    num_cols_readl = min( num_cols_read, n_A - j );
	
	// insert ones into buff_cols where appropriate
	for ( i=0; i < num_cols_readl; i++ ) {
	  buff_cols[ ( i + j ) + ( i ) * ldim_A ] = 1.0;
	}

	// write to A_fp
	err_check = fwrite( buff_cols, sizeof( double ), ldim_A * num_cols_readl, A_fp );			

	// change the ones back into zeroes for the next block
	for ( i=0; i < num_cols_readl; i++ ) {
	  buff_cols[ ( i + j ) + ( i ) * ldim_A ] = 0.0;
	}

  }

  free( buff_cols );

}

// ============================================================================
static int NoFLA_Set_to_identity( int m_A, int n_A, double * buff_A,
               int ldim_A ) {
// Set contents of object A to the identity matrix.
  int  i, j, mn_A;

  // Set the full matrix.
  for ( j = 0; j < n_A; j++ ) {
    for ( i = 0; i < m_A; i++ ) {
      buff_A[ i + j * ldim_A ] = 0.0;
    }
  }
  // Set the main diagonal.
  mn_A = min( m_A, n_A );
  for ( j = 0; j < mn_A; j++ ) {
    buff_A[ j + j * ldim_A ] = 1.0;
  }

  return 0;
}

// ============================================================================
static int NoFLA_Zero_strict_lower_triangular( int m_A, int n_A,
               double * buff_A, int ldim_A ) {
// Zero the strictly lower triangular part of matrix A.
  int  i, j;

  // Set the strictly lower triangular matrix.
  for ( j = 0; j < n_A; j++ ) {
    for ( i = j + 1; i < m_A; i++ ) {
      buff_A[ i + j * ldim_A ] = 0.0;
    }
  }

  return 0;
}

// ============================================================================
static int NoFLA_Zero( int m_A, int n_A, double * buff_A, int ldim_A ) {
// Set the contents of matrix A to zero.
  int  i, j;

  // Set the full matrix.
  for ( j = 0; j < n_A; j++ ) {
    for ( i = 0; i < m_A; i++ ) {
      buff_A[ i + j * ldim_A ] = 0.0;
    }
  }

  return 0;
}

// ============================================================================
static int NoFLA_Copy_vector_into_diagonal( double * v, int m_A, int n_A,
               double * buff_A, int ldim_A ) {
// Copy the contents of vector v into the diagonal of the matrix A.
  int  i, j, mn_A;

  // Copy vector into the diagonal.
  mn_A = min( m_A, n_A );
  for ( i = 0; i < mn_A; i++ ) {
    buff_A[ i + i * ldim_A ] = v[ i ];
  }

  return 0;
}

// ============================================================================
static int NoFLA_Multiply_BAB(
               char transa, char transb,
               int m_A, int n_A, double * buff_A, int ldim_A,
               int m_B, int n_B, double * buff_B, int ldim_B ) {
// Compute B := A * B.

  char    all = 'A';
  double  d_one = 1.0, d_zero = 0.0;
  double  * buff_Bcopy;
  int     ldim_Bcopy;

  if( ( m_B > 0 )&&( n_B > 0 ) ) {
    //// FLA_Obj_create_conf_to( FLA_NO_TRANSPOSE, B, & Bcopy );
    buff_Bcopy = ( double * ) malloc( m_B * n_B * sizeof( double ) );
    ldim_Bcopy = m_B;

    //// FLA_Copy( B, Bcopy );
    dlacpy_( & all, & m_B, & n_B, buff_B, & ldim_B,
                                  buff_Bcopy, & ldim_Bcopy );

    //// FLA_Gemm( transa, transb, FLA_ONE, A, Bcopy, FLA_ZERO, B );
    dgemm_( & transa, & transb, & m_B, & n_B, & m_B,
            & d_one,  buff_A, & ldim_A,
                      buff_Bcopy, & ldim_Bcopy,
            & d_zero, buff_B, & ldim_B );

    //// FLA_Obj_free( & Bcopy );
    free( buff_Bcopy );
  }

  return 0;
}

// ============================================================================
static int NoFLA_Multiply_BBA(
               char transa, char transb,
               int m_A, int n_A, double * buff_A, int ldim_A,
               int m_B, int n_B, double * buff_B, int ldim_B ) {
// Compute B := B * A.

  char    all = 'A';
  double  d_one = 1.0, d_zero = 0.0;
  double  * buff_Bcopy;
  int     ldim_Bcopy;

  if( ( m_B > 0 )&&( n_B > 0 ) ) {
    //// FLA_Obj_create_conf_to( FLA_NO_TRANSPOSE, B, & Bcopy );
    buff_Bcopy = ( double * ) malloc( m_B * n_B * sizeof( double ) );
    ldim_Bcopy = m_B;

    //// FLA_Copy( B, Bcopy );
    dlacpy_( & all, & m_B, & n_B, buff_B, & ldim_B,
                                  buff_Bcopy, & ldim_Bcopy );

    //// FLA_Gemm( transb, transa, FLA_ONE, Bcopy, A, FLA_ZERO, B );
    dgemm_( & transb, & transa, & m_B, & n_B, & n_B,
            & d_one,  buff_Bcopy, & ldim_Bcopy,
                      buff_A, & ldim_A,
            & d_zero, buff_B, & ldim_B );

    //// FLA_Obj_free( & Bcopy );
    free( buff_Bcopy );
  }

  return 0;
}

// ============================================================================
static int NoFLA_Compute_svd(
               int m_A, int n_A, double * buff_A, int ldim_A,
               int m_U, int n_U, double * buff_U, int ldim_U,
               int n_sv, double * buff_sv,
               int m_V, int n_V, double * buff_V, int ldim_V,
               int nb_alg ) {
// Compute:  U, and V of svd of A.
  char    all = 'A';
  double  * buff_Workspace;
  int     info, max_mn_A, min_mn_A, lwork;

  // Some initializations.
  max_mn_A = max( m_A, n_A );
  min_mn_A = min( m_A, n_A );

  // Create Workspace.
  // According to lapack's documentation,
  // workspace for dgebd2 should be: max( m, n ), and
  // workspace for dbdsqr should be: 2*n.
  // However, dgebd2 seems to need more.  So, workspace is increased.
  lwork  = max( 1,
                max( 3 * min_mn_A + max_mn_A, 5 * min_mn_A ) )
           + nb_alg * max_mn_A + 100000 + 10 * m_A + 10 * n_A;
  //// FLA_Obj_create( FLA_Obj_datatype( A ), lwork, 1, 0, 0, & Workspace );
  buff_Workspace = ( double * ) malloc( lwork * sizeof( double ) );
  //// printf( " lwork: %d\n ", lwork );

  // Call to SUBROUTINE DGESVD( JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT,
  //                            WORK, LWORK, INFO )
  dgesvd_( & all, & all, & m_A, & n_A,
           buff_A, & ldim_A, buff_sv,
           buff_U, & ldim_U, buff_V, & ldim_V,
           buff_Workspace, & lwork, & info );
  if( info != 0 ) {
    fprintf( stderr, " *** Info after dgesvd_f: %d \n", info );
  }

  // Remove object Work.
  //// FLA_Obj_free( & Workspace );
  free( buff_Workspace );

  return 0;
}

// ============================================================================
static int NoFLA_Normal_random_matrix( int m_A, int n_A,
               double * buff_A, int ldim_A ) {
//
// It generates a random matrix with normal distribution.
//
  int  i, j;

  // Main loop.
  for ( j = 0; j < n_A; j++ ) {
    for ( i = 0; i < m_A; i++ ) {
      buff_A[ i + j * ldim_A ] = NoFLA_Normal_random_number( 0.0, 1.0 );
    }
  }

  return 0;
}

// ============================================================================
static double NoFLA_Normal_random_number( double mu, double sigma ) {
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

// ============================================================================
static int NoFLA_Apply_Q_WY_lhfc_blk_var2(
               int m_U, int n_U, double * buff_U, int ldim_U,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_B, int n_B, double * buff_B, int ldim_B ) {
//
// It applies the transpose of a block transformation Q to a matrix B from
// the left:
//   B := Q' * B
// where:
//   Q = I - U * T' * U'.
//
  double  * buff_W;
  int     ldim_W;

  // Create auxiliary object.
  //// FLA_Obj_create_conf_to( FLA_NO_TRANSPOSE, B1, & W );
  buff_W = ( double * ) malloc( n_B * n_U * sizeof( double ) );
  ldim_W = max( 1, n_B );

  // Apply the block transformation.
  dlarfb_( "Left", "Transpose", "Forward", "Columnwise",
           & m_B, & n_B, & n_U, buff_U, & ldim_U, buff_T, & ldim_T,
           buff_B, & ldim_B, buff_W, & ldim_W );

  // Remove auxiliary object.
  //// FLA_Obj_free( & W );
  free( buff_W );

  return 0;
}

// ============================================================================
static int NoFLA_Apply_Q_WY_rnfc_blk_var2(
               int m_U, int n_U, double * buff_U, int ldim_U,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_B, int n_B, double * buff_B, int ldim_B ) {
//
// It applies a block transformation Q to a matrix B from the right:
//   B = B * Q
// where:
//   Q = I - U * T' * U'.
//
  double  * buff_W;
  int     ldim_W;

  // Create auxiliary object.
  //// FLA_Obj_create_conf_to( FLA_TRANSPOSE, B1, & W );
  buff_W = ( double * ) malloc( m_B * n_U * sizeof( double ) );
  ldim_W = max( 1, m_B );

  // Apply the block transformation.
  dlarfb_( "Right", "No transpose", "Forward", "Columnwise",
           & m_B, & n_B, & n_U, buff_U, & ldim_U, buff_T, & ldim_T,
           buff_B, & ldim_B, buff_W, & ldim_W );

  // Remove auxiliary object.
  //// FLA_Obj_free( & W );
  free( buff_W );

  return 0;
}

// ============================================================================
static int NoFLA_QRP_WY_unb_var2( int pivoting, int num_stages,
               int m_A, int n_A, double * buff_A, int ldim_A,
               double * buff_T, int ldim_T ) {
  //
  // It computes an unblocked QR factorization of matrix A with or without
  // pivoting. Matrices B and C are optionally pivoted, and matrix T is
  // optionally built.
  //
  // Arguments:
  // "pivoting": If pivoting==1, then QR factorization with pivoting is used.
  // "numstages": It tells the number of columns that are factorized.
  //   If "num_stages" is negative, the whole matrix A is factorized.
  //   If "num_stages" is positive, only the first "num_stages" are factorized.
  int     j, mn_A, m_a21, m_A22, n_A22, n_dB, idx_max_col,
          i_one = 1, n_house_vector, m_rest;
  double  * buff_d, * buff_e, * buff_workspace, * buff_t, diag;
  int     idamax_();

  //// printf( "NoFLA_QRP_WY_unb_var2. pivoting: %d \n", pivoting );

  // Some initializations.
  mn_A    = min( m_A, n_A );

  // Set the number of stages, if needed.
  if( num_stages < 0 ) {
    num_stages = mn_A;
  }

  // Create auxiliary vectors.
  buff_d         = ( double * ) malloc( n_A * sizeof( double ) );
  buff_e         = ( double * ) malloc( n_A * sizeof( double ) );
  buff_workspace = ( double * ) malloc( n_A * sizeof( double ) );
  buff_t         = ( double * ) malloc( n_A * sizeof( double ) );

  if( pivoting == 1 ) {
    // Compute initial norms of A into d and e.
    NoFLA_QRP_compute_norms( m_A, n_A, buff_A, ldim_A, buff_d, buff_e );
  }

  // Main Loop.
  for( j = 0; j < num_stages; j++ ) {
    n_dB  = n_A - j;
    m_a21 = m_A - j - 1;
    m_A22 = m_A - j - 1;
    n_A22 = n_A - j - 1;

    if( pivoting == 1 ) {
      // Obtain the index of the column with largest 2-norm.
      idx_max_col = idamax_( & n_dB, & buff_d[ j ], & i_one ) - 1;

      // Swap columns of A, B, C, pivots, and norms vectors.
      NoFLA_QRP_pivot_G( idx_max_col,
          m_A, & buff_A[ 0 + j * ldim_A ], ldim_A,
          & buff_d[ j ],
          & buff_e[ j ] );
    }

    // Compute tau1 and u21 from alpha11 and a21 such that tau1 and u21
    // determine a Householder transform H such that applying H from the
    // left to the column vector consisting of alpha11 and a21 annihilates
    // the entries in a21 (and updates alpha11).
    n_house_vector = m_a21 + 1;
    dlarfg_( & n_house_vector,
             & buff_A[ j + j * ldim_A ],
             & buff_A[ min( m_A-1, j+1 ) + j * ldim_A ], & i_one,
             & buff_t[ j ] );

    // / a12t \ =  H / a12t \
    // \ A22  /      \ A22  /
    //
    // where H is formed from tau1 and u21.
    diag = buff_A[ j + j * ldim_A ];
    buff_A[ j + j * ldim_A ] = 1.0;
    m_rest = m_A22 + 1;
    dlarf_( "Left", & m_rest, & n_A22,
        & buff_A[ j + j * ldim_A ], & i_one,
        & buff_t[ j ],
        & buff_A[ j + ( j+1 ) * ldim_A ], & ldim_A,
        buff_workspace );
    buff_A[ j + j * ldim_A ] = diag;

    if( pivoting == 1 ) {
      // Update partial column norms.
      NoFLA_QRP_downdate_partial_norms( m_A22, n_A22,
          & buff_d[ j+1 ], 1,
          & buff_e[ j+1 ], 1,
          & buff_A[ j + ( j+1 ) * ldim_A ], ldim_A,
          & buff_A[ ( j+1 ) + min( n_A-1, ( j+1 ) ) * ldim_A ], ldim_A );
    }
  }

  // Build T.
  dlarft_( "Forward", "Columnwise", & m_A, & num_stages, buff_A, & ldim_A,
           buff_t, buff_T, & ldim_T );

  // Remove auxiliary vectors.
  free( buff_d );
  free( buff_e );
  free( buff_workspace );
  free( buff_t );

  return 0;
}

// ============================================================================
static int NoFLA_QRP_compute_norms(
               int m_A, int n_A, double * buff_A, int ldim_A,
               double * buff_d, double * buff_e ) {
//
// It computes the column norms of matrix A. The norms are stored into
// vectors d and e.
//
  int     j, i_one = 1;
  double  dnrm2_();

  // Main loop.
  for( j = 0; j < n_A; j++ ) {
    * buff_d = dnrm2_( & m_A, buff_A, & i_one );
    * buff_e = * buff_d;
    buff_A += ldim_A;
    buff_d++;
    buff_e++;
  }

  return 0;
}

// ============================================================================
static int NoFLA_QRP_downdate_partial_norms( int m_A, int n_A,
               double * buff_d,  int st_d,
               double * buff_e,  int st_e,
               double * buff_wt, int st_wt,
               double * buff_A,  int ldim_A ) {
//
// It updates (downdates) the column norms of matrix A. It uses Drmac's method.
//
  int     j, i_one = 1;
  double  * ptr_d, * ptr_e, * ptr_wt, * ptr_A;
  double  temp, temp2, temp5, tol3z;
  double  dnrm2_(), dlamch_();

  /*
*
*           Update partial column norms
*
          DO 30 J = I + 1, N
             IF( WORK( J ).NE.ZERO ) THEN
*
*                 NOTE: The following 4 lines follow from the analysis in
*                 Lapack Working Note 176.
*
                TEMP = ABS( A( I, J ) ) / WORK( J )
                TEMP = MAX( ZERO, ( ONE+TEMP )*( ONE-TEMP ) )
                TEMP2 = TEMP*( WORK( J ) / WORK( N+J ) )**2
                IF( TEMP2 .LE. TOL3Z ) THEN
                   IF( M-I.GT.0 ) THEN
                      WORK( J ) = DNRM2( M-I, A( I+1, J ), 1 )
                      WORK( N+J ) = WORK( J )
                   ELSE
                      WORK( J ) = ZERO
                      WORK( N+J ) = ZERO
                   END IF
                ELSE
                   WORK( J ) = WORK( J )*SQRT( TEMP )
                END IF
             END IF
 30       CONTINUE
  */

  // Some initializations.
  tol3z = sqrt( dlamch_( "Epsilon" ) );
  ptr_d  = buff_d;
  ptr_e  = buff_e;
  ptr_wt = buff_wt;
  ptr_A  = buff_A;

  // Main loop.
  for( j = 0; j < n_A; j++ ) {
    if( * ptr_d != 0.0 ) {
      temp = dabs( * ptr_wt ) / * ptr_d;
      temp = max( 0.0, ( 1.0 + temp ) * ( 1 - temp ) );
      temp5 = * ptr_d / * ptr_e;
      temp2 = temp * temp5 * temp5;
      if( temp2 <= tol3z ) {
        if( m_A > 0 ) {
          * ptr_d = dnrm2_( & m_A, ptr_A, & i_one );
          * ptr_e = *ptr_d;
        } else {
          * ptr_d = 0.0;
          * ptr_e = 0.0;
        }
      } else {
        * ptr_d = * ptr_d * sqrt( temp );
      }
    }
    ptr_A  += ldim_A;
    ptr_d  += st_d;
    ptr_e  += st_e;
    ptr_wt += st_wt;
  }

  return 0;
}


// ============================================================================
static int NoFLA_QRP_pivot_G( int j_max_col,
               int m_G, double * buff_G, int ldim_G,
               double * buff_d, double * buff_e ) {
//
// It pivots matrix G, pivot vector p, and norms vectors d and e.
// Matrices B and C are optionally pivoted.
//
  int     i_one = 1;
  double  * ptr_g1, * ptr_g2; //// , * ptr_b1, * ptr_b2, * ptr_c1, * ptr_c2;

  // Swap columns of G, pivots, and norms.
  if( j_max_col != 0 ) {

    // Swap full column 0 and column "j_max_col" of G.
    ptr_g1 = & buff_G[ 0 + 0         * ldim_G ];
    ptr_g2 = & buff_G[ 0 + j_max_col * ldim_G ];
    dswap_( & m_G, ptr_g1, & i_one, ptr_g2, & i_one );

    // Copy norms of column 0 to column "j_max_col".
    buff_d[ j_max_col ] = buff_d[ 0 ];
    buff_e[ j_max_col ] = buff_e[ 0 ];
  }

  return 0;
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
