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

Robert van de Geijn
  Dept. of Computer Science and Institute for Computational Engineering and 
  Sciences, 
  The University of Texas at Austin
  Austin, TX.

===============================================================================
Copyright
===============================================================================

Copyright (C) 2016, 
  Universitat Jaume I,
  University of Colorado at Boulder,
  The University of Texas at Austin.

===============================================================================
Disclaimer
===============================================================================

This code is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY EXPRESSED OR IMPLIED.

*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "hqrrp_ooc_left.h"
#include <mkl.h>

// Matrices with dimensions smaller than THRESHOLD_FOR_DGEQPF are processed 
// with LAPACK's routine dgeqpf.
// Matrices with dimensions between THRESHOLD_FOR_DGEQPF and 
// THRESHOLD_FOR_DGEQP3 are processed with LAPACK's routine dgeqp3.
// Matrices with dimensions larger than THRESHOLD_FOR_DGEQP3 are processed 
// with the new HQRRP code.
#define THRESHOLD_FOR_DGEQPF   250
#define THRESHOLD_FOR_DGEQP3  1000


#define PROFILE

// ============================================================================
// Definition of macros.

#define max( a, b )  ( (a) > (b) ? (a) : (b) )
#define min( a, b )  ( (a) > (b) ? (b) : (a) )
#define dabs( a )    ( (a) >= 0.0 ? (a) : -(a) )

// ============================================================================
// Compilation declarations.

#undef CHECK_DOWNDATING_OF_Y

// Declaration of local prototypes.

static int NoFLA_Normal_random_matrix( int m_A, int n_A, 
               double * buff_A, int ldim_A );

static double NoFLA_Normal_random_number( double mu, double sigma );

static int Mult_BA_A_out( int m, int n, int k,
				FILE * A_fp, int ldim_A, double * B_p, int ldim_B,
				double * C_p, int ldim_C,
				int bl_size,
				double * t_read, double * t_write, double * t_seek );

static int dlarft_fc( int n, int k, double * buff_V, int ldim_V,
					  double * buff_tau,
					  double * buff_T, int ldim_T );

static int dlarfb_ltfc( int m_A, int n_A, int k,
						double * buff_V, int ldim_V,
						double * buff_T, int ldim_T,
						double * buff_A, int ldim_A,
						double * buff_work, int ldim_work );

static int Downdate_Y( 
               int m_U11, int n_U11, double * buff_U11, int ldim_U11,
               int m_U21, int n_U21, double * buff_U21, int ldim_U21,
               int m_A12, int n_A12, double * buff_A12, int ldim_A12,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_Y2, int n_Y2, double * buff_Y2, int ldim_Y2,
               int m_G1, int n_G1, double * buff_G1, int ldim_G1,
               int m_G2, int n_G2, double * buff_G2, int ldim_G2 );

static int Apply_Q_WY_lh( 
               int m_U, int n_U, double * buff_U, int ldim_U,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_B, int n_B, double * buff_B, int ldim_B );

static int NoFLA_Apply_Q_WY_rnfc_blk_var4( 
               int m_U, int n_U, double * buff_U, int ldim_U,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_B, int n_B, double * buff_B, int ldim_B );

static int QRP_WY( int pivoting, int num_stages, 
               int m_A, int n_A, double * buff_A, int ldim_A,
               int * buff_p, double * buff_t, 
               int pivot_B, int m_B, int * buff_B, int ldim_B,
               int pivot_C, int m_C, double * buff_C, int ldim_C,
               int build_T, double * buff_T, int ldim_T );

static int NoFLA_QRP_compute_norms(
               int m_A, int n_A, double * buff_A, int ldim_A,
               double * buff_d, double * buff_e );

static int NoFLA_QRP_downdate_partial_norms( int m_A, int n_A,
               double * buff_d,  int st_d,
               double * buff_e,  int st_e,
               double * buff_wt, int st_wt,
               double * buff_A,  int ldim_A );

static int NoFLA_QRP_pivot_G_B_C( int j_max_col,
               int m_G, double * buff_G, int ldim_G, 
               int pivot_B, int m_B, int * buff_B, int ldim_B, 
               int pivot_C, int m_C, double * buff_C, int ldim_C, 
               int * buff_p,
               double * buff_d, double * buff_e );

// ============================================================================
// TODO: temporary function for debugging
void print_double_matrix( char * name, int m_A, int n_A, 
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
static void print_int_vector( char * name, int n_v, int * buff_v ) {
   int  i, j;
 
   printf( "%s = [\n", name );
   for( i = 0; i < n_v; i++ ) {
     printf( "%d\n", buff_v[ i ] );
   }
   printf( "];\n" );
 }
// ============================================================================
int hqrrp_ooc_left( char * A_fname, int m_A, int n_A, int ldim_A,
        int * buff_jpvt, double * buff_tau,
        int nb_alg, int kk, int pp,
		int panel_pivoting, int num_cols_read ) {
//
// HQRRP: It computes the Householder QR with Randomized Pivoting of matrix A.
// This routine is almost compatible with LAPACK's dgeqp3.
// The main difference is that this routine does not manage fixed columns.
//
// Main features:
//   * BLAS-3 based.
//   * Norm downdating method by Drmac.
//   * Downdating for computing Y.
//   * No use of libflame.
//   * Compact WY transformations are used instead of UT transformations.
//   * LAPACK's routine dlarfb is used to apply block transformations.
//
// Arguments:
// ----------
// m_A:            Number of rows of matrix A.
// n_A:            Number of columns of matrix A.
// buff_A:         Address/pointer of/to data in matrix A. Matrix A must be 
//                 stored in column-order.
// ldim_A:         Leading dimension of matrix A.
// buff_jpvt:      Input/output vector with the pivots.
// buff_tau:       Output vector with the tau values of the Householder factors.
// nb_alg:         Block size. 
//                 Usual values for nb_alg are 32, 64, etc.
// pp:             Oversampling size.
//                 Usual values for pp are 5, 10, etc.
// panel_pivoting: If panel_pivoting==1, QR with pivoting is applied to 
//                 factorize the panels of matrix A. Otherwise, QR without 
//                 pivoting is used. Usual value for panel_pivoting is 1.
// num_cols_read:  the number of columns of A to read into RAM at once;
//				   the larger the value, the faster the code will run,
//				   but the max value is determined by system memory
//
  int     m_Y, n_Y, ldim_Y,   
          m_W, n_W, ldim_W,		 
          m_G, n_G, ldim_G,
		  m_A1121, n_A1121, ldim_A1121,
		  m_work, n_work, ldim_work,
		  m_A21, n_A21, ldim_A21,
		  m_A_cols, n_A_cols, ldim_A_cols, // for the "blocks of cols" that we read
		  m_A_mid, n_A_mid, n_A_mid_l, ldim_A_mid,
		  n_A_cols_l,					   // out a bunch
		  m_Q, n_Q, ldim_Q, n_Q_l,
		  m_T, n_T, ldim_T,
		  n_Y_l,
		  m_A1121l, n_A1121l;
  
  double  * buff_Y, * buff_Yl,  // the sampling matrix Y 
		  * buff_Yc, * buff_Ycl,
		  * buff_Y1, * buff_Y2,
		  * buff_W, * buff_Wl,	// the matrix W in a compact WY rep of a HH matrix
								// (Y is the HH matrices)
          * buff_t, * buff_tl,  // tau vector for HH matrices 
          * buff_G, * buff_G1, * buff_G2, // random matrix G
		  * buff_A1121, * buff_A1121l,	  // A1121 = [ A11; A12 ]
		  * buff_A_cols,		// for the "blocks of cols" that we read out a bunch
		  * buff_A_mid,         //  
		  * buff_work,			// for use in lapack functions 
		  * buff_Q,             // for holding the info for the previous HH matrices
		  * buff_T,				// for holding the middle T matrix in a VTV' 
								// representation of a HH matrix
		  * buff_A21;			 
		  // suffix 'c' means "copy," 'l' means "local" (i.e. value for the loop step)
  
  int     * buff_p, * buff_pl; // pivot vectors 
  int     * buff_p_Y, * buff_p_Yl; // pivot vectors determined by Y
  
  int     i, j, k, ii, jj, kk, // TODO: do we use all these?
		  b, last_iter, mn_A;
  double  d_zero = 0.0, d_one = 1.0;
  char    n = 'n', t = 't';
  FILE    * A_fp;
  size_t err_check;
  int nb_alg_l;

#ifdef PROFILE
 struct timespec t1, t2;
 uint64_t diff;
 double t_read = 0.0, t_write = 0.0, t_seek = 0.0;
 double t_init_Y = 0.0, t_qr_Y = 0.0, t_qr_A = 0.0, t_update_A = 0.0, t_downdate_Y = 0.0;
 double t_tot = 0.0;
#endif


  // Executable Statements.
  //// printf( "%% NoFLA_HQRRP_WY_blk_var4.\n" );

  // Check arguments.
  if( m_A < 0 ) {
    fprintf( stderr, 
             "ERROR in NoFLA_HQRRP_WY_blk_var4: m_A is < 0.\n" );
  } if( n_A < 0 ) {
    fprintf( stderr, 
             "ERROR in NoFLA_HQRRP_WY_blk_var4: n_A is < 0.\n" );
  } if( ldim_A < max( 1, m_A ) ) {
    fprintf( stderr, 
             "ERROR in NoFLA_HQRRP_WY_blk_var4: ldim_A is < max( 1, m_A ).\n" );
  }

  // Some initializations.
  mn_A   = min( m_A, n_A );
  buff_p = buff_jpvt;
  buff_t = buff_tau;

  A_fp = fopen( A_fname, "r+" );

  // Quick return.
  if( mn_A == 0 ) {
    return 0;
  }

  // Initialize the seed for the generator of random numbers.
  srand( 12 );

  // TODO: do we need all these objects still, for this left-looking algorithm?

  // Create auxiliary objects.
  m_Y     = nb_alg + pp;
  n_Y     = n_A;
  buff_Y  = ( double * ) malloc( m_Y * n_Y * sizeof( double ) );
  buff_Yc  = ( double * ) malloc( m_Y * n_Y * sizeof( double ) );
  ldim_Y  = m_Y;

  m_W     = nb_alg;
  n_W     = n_A;
  buff_W  = ( double * ) malloc( m_W * n_W * sizeof( double ) );
  ldim_W  = m_W;

  m_G     = nb_alg + pp;
  n_G     = m_A;
  buff_G  = ( double * ) malloc( m_G * n_G * sizeof( double ) );
  ldim_G  = m_G;

  m_A1121    = m_A;
  n_A1121    = nb_alg;
  buff_A1121 = ( double * ) malloc( m_A1121 * n_A1121 * sizeof( double ) );
  if ( !buff_A1121 ) {
    printf("Error! Memory allocation for A1121 failed \n");
	return -1;
  }
  ldim_A1121 = m_A;

  m_A_cols = m_A;
  n_A_cols = num_cols_read;
  buff_A_cols = ( double * ) malloc( m_A_cols * n_A_cols * sizeof( double ) );
  ldim_A_cols = m_A;

  m_A_mid = m_A;
  n_A_mid = nb_alg;
  buff_A_mid = ( double * ) malloc( m_A_mid * n_A_mid * sizeof( double ) );
  ldim_A_mid = m_A;

  m_A21    = nb_alg;
  n_A21    = n_A;
  buff_A21 = ( double * ) malloc( m_A21 * n_A21 * sizeof( double ) );
  ldim_A21 =  nb_alg;
  
  m_work    = max( m_A, n_A ) ;
  n_work    = nb_alg;
  buff_work = ( double * ) malloc( m_work * n_work * sizeof( double ) ); 
  ldim_work = max( m_A, n_A );

  m_Q    = m_A;
  n_Q    = num_cols_read;
  buff_Q = ( double * ) malloc( m_Q * n_Q * sizeof( double ) ); 
  ldim_Q = m_A;

  m_T    = nb_alg;
  n_T    = nb_alg;
  buff_T = ( double * ) malloc( m_T * n_T * sizeof( double ) ); 
  ldim_T = nb_alg; 

  buff_p_Y = ( int * ) malloc( m_A * sizeof( int ) );

  // Initialize matrices G and Y.
  // TODO: remove ability to oversample, b/c it isn't implemented everywhere
  NoFLA_Normal_random_matrix( nb_alg + pp, m_A, buff_G, ldim_G );
 
#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t1 );
#endif

  fseek( A_fp, 0, SEEK_SET );

  Mult_BA_A_out( m_Y, n_Y, m_A,
				A_fp, ldim_A, buff_G, ldim_G,
				buff_Y, ldim_Y,
				nb_alg,
				& t_read, & t_write, & t_seek );

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t2 );
  diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
  t_init_Y += ( double ) diff / (1E9);
#endif

  // Main Loop.
  for ( j = 0; j < mn_A; j += nb_alg ) {
    b = min( nb_alg, min( n_A - j, m_A - j ) );

    // Check whether it is the last iteration.
    last_iter = ( ( ( j + nb_alg >= m_A )||( j + nb_alg >= n_A ) ) ? 1 : 0 );

    // Some initializations for the iteration of this loop.
    n_Y_l = n_Y - j;
    buff_Ycl = & buff_Yc[ 0 + j * ldim_Y ];
    buff_Yl = & buff_Y[ 0 + j * ldim_Y ];
    buff_pl = & buff_p[ j ];
    buff_tl = & buff_t[ j ];

    buff_Y1   = & buff_Y[ 0 + j * ldim_Y ];
    buff_Wl = & buff_W[ 0 + j * ldim_W ];

	m_A1121l = m_A - j;
	n_A1121l = b;
	buff_A1121l = & buff_A1121[ j + 0 * ldim_A1121 ];

    m_A21 = nb_alg;
	n_A21 = n_A - j;

    m_G = nb_alg;
	n_G = m_A - j;

	n_A_mid_l = b;

    buff_Y2 = & buff_Y[ 0 + min( n_Y - 1, j + b ) * ldim_Y ];
    buff_G1 = & buff_G[ 0 + j * ldim_G ];
    buff_G2 = & buff_G[ 0 + min( n_G - 1, j + b ) * ldim_G ];

    buff_p_Yl = & buff_p_Y[ j ];
	for ( i=0; i < n_A - j; i++ ) {
	  buff_p_Yl[ i ] = i;
	}
      
    if( last_iter == 0 ) {
	  // generate G
      NoFLA_Normal_random_matrix( nb_alg, m_A - j, buff_G, ldim_G );

      // generate YR
	  for ( i=j; i<n_A; i+= num_cols_read ) {	
	    
	    n_A_cols_l = min( num_cols_read, n_A - i );	

	    // read out block of cols of A, A_i
		fseek( A_fp, ( 0 + ( i ) * ldim_A ) * sizeof( double ), SEEK_SET );	
		err_check = fread( buff_A_cols, sizeof( double ), m_A_cols * n_A_cols_l, A_fp );
	    if ( err_check != m_A_cols * n_A_cols_l ) {
		  printf( "Error! Read of block of A for building of Y failed!\n" );
		  return 1;
		}

		// update the block A_i
		for ( k=0; k < j; k += num_cols_read ) {

		  n_Q_l = min( num_cols_read, j - k );

		  // read out block containing HH vectors
		  fseek( A_fp, ( 0 + ( k ) * ldim_A ) * sizeof( double ), SEEK_SET );	
		  err_check = fread( buff_Q, sizeof( double ), m_Q * n_Q_l, A_fp );
		  if ( err_check != m_Q * n_Q_l ) {
			printf( "Error! Read of block of Q for building of Y failed!\n" );
			return 1;
		  }

		  // apply HH vectors to A_i
		  for ( ii=0; ii<n_Q_l; ii+=nb_alg  ) {
		   
		    nb_alg_l = min( nb_alg, n_Q_l - ii );

			dlarft_fc( m_A - k - ii, nb_alg_l,
					   & buff_Q[ ( k + ii ) + ( ii ) * ldim_Q ], ldim_Q,
					   & buff_tau[ k + ii ],
					   buff_T, ldim_T );

			dlarfb_ltfc( m_A_cols - k - ii, n_A_cols_l, nb_alg_l,
						 & buff_Q[ ( k + ii ) + ( ii ) * ldim_Q ], ldim_Q,
						 buff_T, ldim_T,
						 & buff_A_cols[ ( k + ii ) + ( 0 ) * ldim_A_cols ], ldim_A_cols,
						 buff_work, ldim_work );

		  } // for ii

		} // for k


		// compute a block of cols of Y, Y_i = G * A_i
		dgemm( & n, & n, & m_G, & n_A_cols_l, & n_G,
			   & d_one,
			   buff_G, & ldim_G,
			   & buff_A_cols[ j + ( 0 ) * ldim_A_cols ], & ldim_A_cols,
			   & d_zero,
			   & buff_Y[ 0 + ( i - j ) * ldim_Y ], & ldim_Y );

	  } // for i
	  
      // Compute QRP of Y.
      QRP_WY( 1, b,
          m_Y, n_Y_l, buff_Y, ldim_Y, buff_p_Yl, buff_tl,
          1, 1, buff_pl, 1,
          0, 0, NULL, 0,
          0, NULL, 0 );

	} // if (last_iter == 0)

    // Update panel [ A12; A22; A32 ]
    // Compute QRP.
    // Apply same permutations to A12, and build T1_T.
    //

	  // read out block of matrix [ A12; A22; A32 ]; apply the permutation as we read unless we're on the last iteration 
	  if ( last_iter == 0 ) {
		for ( i=0; i < min( nb_alg, n_A - j ); i++ ) {
		  // read out column from original position to be processed
		  fseek( A_fp, ( j + ( buff_p_Yl[ i ] ) ) * ldim_A * 
				 sizeof( double ), SEEK_SET );
		  err_check = fread( & buff_A_mid[ 0 + i * ldim_A_mid ],
							 sizeof( double ), m_A, A_fp );
		  if ( err_check != m_A ) {
		    printf( "Error! Read of to-be-processed col of A failed!" );
			return 1;
		  }
		  // read out column from original position to be swapped 
		  fseek( A_fp, ( j + i ) * ldim_A * 
				 sizeof( double ), SEEK_SET );
		  err_check = fread( & buff_A_cols[ 0 + i * ldim_A_cols ],
							 sizeof( double ), m_A, A_fp );
		  if ( err_check != m_A ) {
		    printf( "Error! Read of to-be-permuted col of A failed!" );
			return 1;
		  }
		} // for i
		for ( i=0; i < min( nb_alg, n_A - j ); i++ ) {
		  // find new location for column
		  int found = 0, new_ind = 0;
		  while ( found == 0 && new_ind < n_A ) {
		    if ( buff_p_Yl[ new_ind ] == i ) {
			  found = 1;
			} else {
		      new_ind++;
			}
		  }
		  // write out swapped column to new position for later processing
		  fseek( A_fp, ( j + new_ind ) * ldim_A * 
				 sizeof( double ), SEEK_SET );
		  fwrite( & buff_A_cols[ 0 + i * ldim_A_cols ], sizeof( double ), m_A, A_fp );
	    } // for i
	  } else { // just read out last cols of A; nothing needs to be swapped 
		fseek( A_fp, ( j ) * ldim_A * 
			   sizeof( double ), SEEK_SET );
		err_check = fread( buff_A_mid,
						   sizeof( double ), m_A * b, A_fp );
		if ( err_check != m_A * b ) {
		  printf( "Error! Read of to-be-processed block of A failed!" );
		  return 1;
		}

	  } // if

      // update the entire block of cols by applying Q to it
	  for ( i=0; i < j; i += num_cols_read ) {

		n_Q_l = min( num_cols_read, j - i );

		// read out block containing HH vectors
		fseek( A_fp, ( 0 + ( i ) * ldim_A ) * sizeof( double ), SEEK_SET );	
		err_check = fread( buff_Q, sizeof( double ), m_Q * n_Q_l, A_fp );
		if ( err_check != m_Q * n_Q_l ) {
		  printf( "Error! Read of block of Q for building of Y failed!\n" );
		  return 1;
		}

		// apply HH vectors to A_mid
		for ( k=0; k<n_Q_l; k+=nb_alg  ) {
		 
		  nb_alg_l = min( nb_alg, n_Q_l - k );

		  dlarft_fc( m_A - i - k, nb_alg_l,
					 & buff_Q[ ( i + k ) + ( k ) * ldim_Q ], ldim_Q,
					 & buff_tau[ i + k ],
					 buff_T, ldim_T );

		  dlarfb_ltfc( m_A_mid - i - k, n_A_mid_l, nb_alg_l,
					   & buff_Q[ ( i + k ) + ( k ) * ldim_Q ], ldim_Q,
					   buff_T, ldim_T,
					   & buff_A_mid[ ( i + k ) + ( 0 ) * ldim_A_mid ], ldim_A_mid,
					   buff_work, ldim_work );

		} // for k

	  } // for i

	  // compute QRP;
	  QRP_WY( panel_pivoting, -1,
        m_A_mid - j, n_A_mid_l, 
		& buff_A_mid[ j + ( 0 ) * ldim_A_mid ], ldim_A_mid, 
		buff_pl, buff_tl,
        0, 0, NULL, 0,
        1, j, buff_A_mid, ldim_A_mid,
        0, NULL, 0 );
      
	  // write out results of above QRP
	  fseek( A_fp, ( j ) * ldim_A * 
			 sizeof( double ), SEEK_SET );
	  fwrite( buff_A_mid, sizeof( double ), m_A_mid * n_A_mid_l, A_fp );

  } // for j

#ifdef PROFILE
  t_tot += t_read + t_write + t_init_Y + t_qr_Y +
			t_qr_A + t_update_A + t_downdate_Y;

  printf( "%% t_read:          %le    %.2f%%\n", t_read, t_read / t_tot * 100.0 );
  printf( "%% t_write:         %le    %.2f%%\n", t_write, t_write / t_tot * 100.0 );
  printf( "%% t_init_Y:        %le    %.2f%%\n", t_init_Y, t_init_Y / t_tot * 100.0 );
  printf( "%% t_qr_Y:          %le    %.2f%%\n", t_qr_Y, t_qr_Y / t_tot * 100.0 );
  printf( "%% t_qr_A:          %le    %.2f%%\n", t_qr_A, t_qr_A / t_tot * 100.0 );
  printf( "%% t_update_A:      %le    %.2f%%\n", t_update_A, t_update_A / t_tot * 100.0 );
  printf( "%% t_downdate_Y:    %le    %.2f%%\n", t_downdate_Y, t_downdate_Y / t_tot * 100.0 );
  printf( "%% total_time:          %le\n", t_tot );
#endif


  // Remove auxiliary objects.
  fclose( A_fp );

  free( buff_G );
  free( buff_Y );
  free( buff_Yc );
  free( buff_W );
  free( buff_A1121 );
  free( buff_A21 );
  free( buff_A_cols );
  free( buff_A_mid );
  free( buff_work );
  free( buff_Q );
  free( buff_T );
  free( buff_p_Y );

  return 0;
}


// ============================================================================
static int Mult_BA_A_out( int m, int n, int k,
				FILE * A_fp, int ldim_A, double * B_p, int ldim_B,
				double * C_p, int ldim_C,
				int bl_size,
				double * t_read, double * t_write, double * t_seek ) {
  // Computes C <-- B*A when matrix A is stored out of core,
  // and B and C can be stored in core
  // NOTE: the file position of the stream must be set correctly before entry!

  // bl_size is the number of cols of A that can be stored in RAM at a time

  // declare aux vars
  double * A_bl_p; // stores a block of cols of A
  int num_cols;
  size_t check;

  double d_one = 1.0, d_zero = 0.0;
  int i,j;

#ifdef PROFILE
  struct timespec t1, t2;
  uint64_t diff;
#endif

  // some initializations
  A_bl_p = ( double * ) malloc( k * bl_size * sizeof( double ) );

  // do multiplication one block at a time
  for ( i=0; i < n; i+= bl_size ) {
    
	num_cols = min( bl_size, n - i );
	
	// read a block of A into memory, one col at a time

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t1 );
#endif
    
	for ( j=0; j<num_cols; j++ ) {
	  
	  check = fread( & A_bl_p[ 0 + j * k ], sizeof( double ), k, A_fp ); 
	  if ( ( int ) check != k ) {
	    printf( "Warning! read failed in Mult_BA_A_out. check = %d \n", (int)check );  
	    return 1;
	  }
	
	  // position file pointer for next iteration
	  fseek( A_fp, ( ldim_A - k ) * sizeof( double ), SEEK_CUR );

	}

#ifdef PROFILE
  clock_gettime( CLOCK_MONOTONIC, & t2 );
  diff = (1E9) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
  * t_read += ( double ) diff / (1E9);
#endif

	// do multiplication; gives one block of cols of C
	dgemm( "No transpose", "No tranpose", 
			& m, & num_cols, & k,
			& d_one, B_p, & ldim_B, A_bl_p, & k,
			& d_zero, & C_p[ 0 + i * ldim_C ], & ldim_C );
  
  }

  // free memory
  free( A_bl_p );

  return 0;

}

// ============================================================================
static int dlarft_fc( int n, int k, double * buff_V, int ldim_V,
					  double * buff_tau,
					  double * buff_T, int ldim_T ) {
// performs the same function as the LAPACK dlarft, with parameters
// direct = "F",
// storev = "C"
  
  char f = 'F', c = 'C';

  dlarft( & f, & c,
		  & n, & k, 
		  buff_V, & ldim_V,
		  buff_tau,
		  buff_T, & ldim_T );
  
  return 0;

}
// ============================================================================
static int dlarfb_ltfc( int m_A, int n_A, int k,
						double * buff_V, int ldim_V,
						double * buff_T, int ldim_T,
						double * buff_A, int ldim_A,
						double * buff_work, int ldim_work ) {
// carries out the lapack function dlarfb, with parameters
// side = "L",
// trans = "T",
// direct = "F",
// storev = "C"

  char l = 'L', t = 'T', f = 'F', c = 'C';

  dlarfb( & l, & t, & f, & c,
		  & m_A, & n_A, & k, 
		  buff_V, & ldim_V, buff_T, & ldim_T,
		  buff_A, & ldim_A,
		  buff_work, & ldim_work );

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

/* ========================================================================= */
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
static int Downdate_Y( 
               int m_U11, int n_U11, double * buff_U11, int ldim_U11,
               int m_U21, int n_U21, double * buff_U21, int ldim_U21,
               int m_A12, int n_A12, double * buff_A12, int ldim_A12,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_Y2, int n_Y2, double * buff_Y2, int ldim_Y2,
               int m_G1, int n_G1, double * buff_G1, int ldim_G1,
               int m_G2, int n_G2, double * buff_G2, int ldim_G2 ) {
//
// It downdates matrix Y, and updates matrix G.
// Only Y2 of Y is updated.
// Only G1 and G2 of G are updated.
//
// Y2 = Y2 - ( G1 - ( G1*U11 + G2*U21 ) * T11 * U11' ) * R12.
//
  int    i, j;
  double * buff_B;
  double d_one       = 1.0;
  double d_minus_one = -1.0;
  int    m_B         = m_G1;
  int    n_B         = n_G1;
  int    ldim_B      = m_G1;

  // Create object B.
  //// FLA_Obj_create_conf_to( FLA_NO_TRANSPOSE, G1, & B );
  buff_B = ( double * ) malloc( m_B * n_B * sizeof( double ) );

  // B = G1.
  //// FLA_Copy( G1, B );
  dlacpy_( "All", & m_G1, & n_G1, buff_G1, & ldim_G1,
                                  buff_B, & ldim_B );

  // B = B * U11.
  //// FLA_Trmm( FLA_RIGHT, FLA_LOWER_TRIANGULAR,
  ////           FLA_NO_TRANSPOSE, FLA_UNIT_DIAG,
  ////           FLA_ONE, U11, B );
  dtrmm( "Right", "Lower", "No transpose", "Unit", & m_B, & n_B,
          & d_one, buff_U11, & ldim_U11, buff_B, & ldim_B );

  // B = B + G2 * U21.
  //// FLA_Gemm( FLA_NO_TRANSPOSE, FLA_NO_TRANSPOSE,
  ////           FLA_ONE, G2, U21, FLA_ONE, B );
  dgemm( "No transpose", "No tranpose", & m_B, & n_B, & m_U21,
          & d_one, buff_G2, & ldim_G2, buff_U21, & ldim_U21,
          & d_one, buff_B, & ldim_B );

  // B = B * T11.
  //// FLA_Trsm( FLA_RIGHT, FLA_UPPER_TRIANGULAR,
  ////           FLA_NO_TRANSPOSE, FLA_NONUNIT_DIAG,
  ////           FLA_ONE, T, B );
  //// dtrsm_( "Right", "Upper", "No transpose", "Non-unit", & m_B, & n_B,
  ////         & d_one, buff_T, & ldim_T, buff_B, & ldim_B );
  // Used dtrmm instead of dtrsm because of using compact WY instead of UT.
  dtrmm( "Right", "Upper", "No transpose", "Non-unit", & m_B, & n_B,
          & d_one, buff_T, & ldim_T, buff_B, & ldim_B );

  // B = - B * U11^H.
  //// FLA_Trmm( FLA_RIGHT, FLA_LOWER_TRIANGULAR,
  ////           FLA_CONJ_TRANSPOSE, FLA_UNIT_DIAG,
  ////           FLA_MINUS_ONE, U11, B );
  dtrmm( "Right", "Lower", "Conj_tranpose", "Unit", & m_B, & n_B,
          & d_minus_one, buff_U11, & ldim_U11, buff_B, & ldim_B );

  // B = G1 + B.
  //// FLA_Axpy( FLA_ONE, G1, B );
  for( j = 0; j < n_B; j++ ) {
    for( i = 0; i < m_B; i++ ) {
      buff_B[ i + j * ldim_B ] += buff_G1[ i + j * ldim_G1 ];
    }
  }

  // Y2 = Y2 - B * R12.
  //// FLA_Gemm( FLA_NO_TRANSPOSE, FLA_NO_TRANSPOSE,
  ////           FLA_MINUS_ONE, B, A12, FLA_ONE, Y2 );
  dgemm( "No transpose", "No transpose", & m_Y2, & n_Y2, & m_A12,
          & d_minus_one, buff_B, & ldim_B, buff_A12, & ldim_A12,
          & d_one, buff_Y2, & ldim_Y2 );

  //
  // GR = GR * Q
  //
  NoFLA_Apply_Q_WY_rnfc_blk_var4( 
          m_U11 + m_U21, n_U11, buff_U11, ldim_U11,
          m_T, n_T, buff_T, ldim_T,
          m_G1, n_G1 + n_G2, buff_G1, ldim_G1 );

  // Remove object B.
  //// FLA_Obj_free( & B );
  free( buff_B );

  return 0;
}

// ============================================================================
static int Apply_Q_WY_lh( 
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
static int NoFLA_Apply_Q_WY_rnfc_blk_var4( 
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
static int QRP_WY( int pivoting, int num_stages, 
               int m_A, int n_A, double * buff_A, int ldim_A,
               int * buff_p, double * buff_t, 
               int pivot_B, int m_B, int * buff_B, int ldim_B,
               int pivot_C, int m_C, double * buff_C, int ldim_C,
               int build_T, double * buff_T, int ldim_T ) {
//
// It computes an unblocked QR factorization of matrix A with or without 
// pivoting. Matrices B and C are optionally pivoted, and matrix W is
// optionally built.
//
// Arguments:
// "pivoting": If pivoting==1, then QR factorization with pivoting is used.
// "numstages": It tells the number of columns that are factorized.
//   If "num_stages" is negative, the whole matrix A is factorized.
//   If "num_stages" is positive, only the first "num_stages" are factorized.
// "pivot_B": if "pivot_B" is true, matrix "B" is pivoted too.
// "pivot_C": if "pivot_C" is true, matrix "C" is pivoted too.
// "build_T": if "build_T" is true, matrix "T" is built.
//
  int     j, mn_A, m_a21, m_A22, n_A22, n_dB, idx_max_col, 
          i_one = 1, n_house_vector, m_rest;
  double  * buff_d, * buff_e, * buff_workspace, diag;
  int     idamax_();

  //// printf( "QRP_WY. pivoting: %d \n", pivoting );

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
      NoFLA_QRP_pivot_G_B_C( idx_max_col,
          m_A, & buff_A[ 0 + j * ldim_A ], ldim_A,
          pivot_B, m_B, & buff_B[ 0 + j * ldim_B ], ldim_B,
          pivot_C, m_C, & buff_C[ 0 + j * ldim_C ], ldim_C,
          & buff_p[ j ],
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
  if( build_T ) {
    dlarft_( "Forward", "Columnwise", & m_A, & num_stages, buff_A, & ldim_A, 
             buff_t, buff_T, & ldim_T );
  }

  // Remove auxiliary vectors.
  free( buff_d );
  free( buff_e );
  free( buff_workspace );

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
static int NoFLA_QRP_pivot_G_B_C( int j_max_col,
               int m_G, double * buff_G, int ldim_G, 
               int pivot_B, int m_B, int * buff_B, int ldim_B, 
               int pivot_C, int m_C, double * buff_C, int ldim_C, 
               int * buff_p,
               double * buff_d, double * buff_e ) {
//
// It pivots matrix G, pivot vector p, and norms vectors d and e.
// Matrices B and C are optionally pivoted.
//
  int     ival, i_one = 1;
  double  * ptr_g1, * ptr_g2, * ptr_c1, * ptr_c2;
  int     btemp;

  // Swap columns of G, pivots, and norms.
  if( j_max_col != 0 ) {

    // Swap full column 0 and column "j_max_col" of G.
    ptr_g1 = & buff_G[ 0 + 0         * ldim_G ];
    ptr_g2 = & buff_G[ 0 + j_max_col * ldim_G ];
    dswap( & m_G, ptr_g1, & i_one, ptr_g2, & i_one );

    // Swap full column 0 and column "j_max_col" of B.
    if( pivot_B ) {
	  btemp = buff_B[ 0 + 0 * ldim_B ]; 
      buff_B[ 0 + 0 * ldim_B ] = buff_B[ 0 + j_max_col * ldim_B ];
	  buff_B[ 0 + j_max_col * ldim_B ] = btemp;
    }

    // Swap full column 0 and column "j_max_col" of C.
    if( pivot_C ) {
      ptr_c1 = & buff_C[ 0 + 0         * ldim_C ];
      ptr_c2 = & buff_C[ 0 + j_max_col * ldim_C ];
      dswap( & m_C, ptr_c1, & i_one, ptr_c2, & i_one );
    }

    // Swap element 0 and element "j_max_col" of pivot vector "p".
    ival = buff_p[ j_max_col ];
    buff_p[ j_max_col ] = buff_p[ 0 ];
    buff_p[ 0 ] = ival;

    // Copy norms of column 0 to column "j_max_col".
    buff_d[ j_max_col ] = buff_d[ 0 ];
    buff_e[ j_max_col ] = buff_e[ 0 ];
  }

  return 0;
}

