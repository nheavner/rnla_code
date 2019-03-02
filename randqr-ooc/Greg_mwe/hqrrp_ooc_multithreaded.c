#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "hqrrp_ooc.h"
#include <mkl.h>
#include <omp.h>

// Matrices with dimensions smaller than THRESHOLD_FOR_DGEQPF are processed 
// with LAPACK's routine dgeqpf.
// Matrices with dimensions between THRESHOLD_FOR_DGEQPF and 
// THRESHOLD_FOR_DGEQP3 are processed with LAPACK's routine dgeqp3.
// Matrices with dimensions larger than THRESHOLD_FOR_DGEQP3 are processed 
// with the new HQRRP code.
#define THRESHOLD_FOR_DGEQPF   250
#define THRESHOLD_FOR_DGEQP3  1000


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
				int bl_size, int num_cols_read );

static int Downdate_Y( 
               int m_A12, int n_A12, double * buff_A12, int ldim_A12,
               int m_Y2, int n_Y2, double * buff_Y2, int ldim_Y2,
			   int m_Y_aux, int n_Y_aux, double * buff_Y_aux, int ldim_Y_aux );

static int Form_Y_aux( 
               int m_U11, int n_U11, double * buff_U11, int ldim_U11,
               int m_U21, int n_U21, double * buff_U21, int ldim_U21,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_G1, int n_G1, double * buff_G1, int ldim_G1,
               int m_G2, int n_G2, double * buff_G2, int ldim_G2,
			   int m_Y_aux, int n_Y_aux, double * buff_Y_aux, int ldim_Y_aux );

static int Downdate_G( 
               int m_U11, int n_U11, double * buff_U11, int ldim_U11,
               int m_U21, int n_U21, double * buff_U21, int ldim_U21,
               int m_T, int n_T, double * buff_T, int ldim_T,
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

static struct timespec start_timer( void );

static double stop_timer( struct timespec t1 );

// ============================================================================
int hqrrp_ooc_multithreaded( char * dir_name, size_t dir_name_size, char * A_fname, size_t A_fname_size, 
		int m_A, int n_A, int ldim_A,
        int * buff_jpvt, double * buff_tau,
        int nb_alg, int kk, int pp, int panel_pivoting ) {
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
// kk:             Gives number of columns to process.
//                 If k == n, a full factorization is computed.
//                 If k < n, then floor(k / nb_alg) columns are processed
// pp:             Oversampling size.
//                 Usual values for pp are 5, 10, etc.
// panel_pivoting: If panel_pivoting==1, QR with pivoting is applied to 
//                 factorize the panels of matrix A. Otherwise, QR without 
//                 pivoting is used. Usual value for panel_pivoting is 1.
//
  int     m_Y, n_Y, ldim_Y,   
          m_W, n_W, ldim_W,		 
          m_G, n_G, ldim_G,
		  m_A1112, n_A1112, ldim_A1112,
		  m_A_mid, n_A_mid, ldim_A_mid,
		  m_A_mid_old, n_A_mid_old, ldim_A_mid_old,
		  m_work, n_work_next, n_work_current, ldim_work,
		  m_A21, n_A21, ldim_A21,
		  n_Ycl,
		  m_Y_updt_aux, n_Y_updt_aux, ldim_Y_updt_aux,
		  m_A1112l, n_A1112l;
  
  double  * buff_Y, * buff_Yl,  // the sampling matrix Y 
		  * buff_Yc, * buff_Ycl,
		  * buff_Y1, * buff_Y2,
		  * buff_Y_updt_aux,
		  * buff_W, * buff_Wl,	// the matrix W in a compact WY rep of a HH matrix
								// (Y is the HH matrices)
          * buff_t, * buff_tl,  // tau vector for HH matrices 
          * buff_G, * buff_G1, * buff_G2, // random matrix G
		  * buff_A_mid,         // A_mid = [A12; A22; A32]
		  * buff_A_mid_old,     // A_mid before the permutations of the current loop had been
								// applied
		  * buff_A1112, * buff_A1112l,	  // A1112 = [ A11; A12 ]
		  * buff_work_current, * buff_work_next,  // for holding chunks of cols of A for applying 
												// updates
		  * buff_A21;			 
		  // suffix 'c' means "copy," 'l' means "local" (i.e. value for the loop step)
  
  int     * buff_p, * buff_pl; // pivot vectors
  int	  * buff_p_Y, * buff_p_Yl; // pivots determined by Y
  int     * buff_p_bl; // pivots for the tall thin block
  
  int     i, j, k,
		  b, last_iter, mn_A;
  double  d_zero = 0.0, d_one = 1.0;
  int     i_one = 1;
  FILE    * A_fp;
  size_t err_check;
  int num_cols_read = 5000;

  // set maximum number of threads 
  omp_set_num_threads( omp_get_num_procs( ) );


  // Executable Statements.

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

  char file_path[ dir_name_size / sizeof( dir_name[0] ) + 
		A_fname_size / sizeof( A_fname[0] ) ];
  strcpy( file_path, dir_name );
  strcat( file_path, A_fname );

  A_fp = fopen( file_path, "r+" );

  // Quick return.
  if( mn_A == 0 ) {
    return 0;
  }

  // Initialize the seed for the generator of random numbers.
  srand( 12 );

  // Create auxiliary objects.
  m_Y     = nb_alg + pp;
  n_Y     = n_A;
  buff_Y  = ( double * ) malloc( m_Y * n_Y * sizeof( double ) );
  buff_Yc  = ( double * ) malloc( m_Y * n_Y * sizeof( double ) );
  ldim_Y  = m_Y;

  m_Y_updt_aux = nb_alg;
  n_Y_updt_aux = nb_alg;
  buff_Y_updt_aux = ( double * ) malloc( m_Y_updt_aux * n_Y_updt_aux * sizeof( double ) );
  ldim_Y_updt_aux = m_Y_updt_aux;

  m_W     = nb_alg;
  n_W     = n_A;
  buff_W  = ( double * ) malloc( m_W * n_W * sizeof( double ) );
  ldim_W  = m_W;

  m_G     = nb_alg + pp;
  n_G     = m_A;
  buff_G  = ( double * ) malloc( m_G * n_G * sizeof( double ) );
  ldim_G  = m_G;
  
  m_A_mid    = m_A;
  n_A_mid    = nb_alg;
  buff_A_mid = ( double * ) malloc( m_A_mid * n_A_mid * sizeof( double ) );
  ldim_A_mid = m_A_mid;

  m_A_mid_old    = m_A;
  n_A_mid_old    = nb_alg;
  buff_A_mid_old = ( double * ) malloc( m_A_mid_old * n_A_mid_old * sizeof( double ) );
  ldim_A_mid_old = m_A_mid_old;
  
  m_A1112    = m_A;
  n_A1112    = nb_alg;
  buff_A1112 = ( double * ) malloc( m_A1112 * n_A1112 * sizeof( double ) );
  if ( !buff_A1112 ) {
    printf("Error! Memory allocation for A1112 failed \n");
	return -1;
  }
  ldim_A1112 = m_A;

  m_A21    = nb_alg;
  n_A21    = n_A;
  buff_A21 = ( double * ) malloc( m_A21 * n_A21 * sizeof( double ) );
  ldim_A21 =  nb_alg;
  
  m_work    = m_A;
  n_work_current    = num_cols_read;
  buff_work_current = ( double * ) malloc( m_work * n_work_current * sizeof( double ) ); 
  buff_work_next = ( double * ) malloc( m_work * n_work_current * sizeof( double ) ); 
  ldim_work = m_A;

  buff_p_Y  = ( int * ) malloc( m_A * sizeof( int ) ); 
  buff_p_bl  = ( int * ) malloc( nb_alg * sizeof( int ) ); 

  // Initialize matrices G and Y.
  /*
  NoFLA_Normal_random_matrix( nb_alg + pp, m_A, buff_G, ldim_G );
 
  fseek( A_fp, 0, SEEK_SET );

  Mult_BA_A_out( m_Y, n_Y, m_A,
				A_fp, ldim_A, buff_G, ldim_G,
				buff_Y, ldim_Y,
				nb_alg, num_cols_read );
  */

  // Main Loop.
  for( j = 0; j < kk; j += nb_alg ) {
    b = min( nb_alg, min( n_A - j, m_A - j ) );

    // Check whether it is the last iteration.
    last_iter = ( ( ( j + nb_alg >= m_A )||( j + nb_alg >= n_A ) ) ? 1 : 0 );

    // Some initializations for the iteration of this loop.
    n_Ycl = n_Y - j;
    buff_Ycl = & buff_Yc[ 0 + j * ldim_Y ];
    buff_Yl = & buff_Y[ 0 + j * ldim_Y ];
    buff_pl = & buff_p[ j ];
    buff_tl = & buff_t[ j ];

    buff_Y1   = & buff_Y[ 0 + j * ldim_Y ];
    buff_Wl = & buff_W[ 0 + j * ldim_W ];

    m_A_mid = m_A;
	n_A_mid = min( nb_alg, n_A - j );

    m_A_mid_old = m_A;
	n_A_mid_old = min( nb_alg, n_A - j );

	m_A1112l = m_A - j;
	n_A1112l = b;
	buff_A1112l = & buff_A1112[ j + 0 * ldim_A1112 ];

    m_A21 = nb_alg;
	n_A21 = n_A - j;

    buff_Y2 = & buff_Y[ 0 + min( n_Y - 1, j + b ) * ldim_Y ];
    buff_G1 = & buff_G[ 0 + j * ldim_G ];
    buff_G2 = & buff_G[ 0 + min( n_G - 1, j + b ) * ldim_G ];
      
	buff_p_Yl = & buff_p_Y[ j ];
	for ( i=0; i < n_A - j; i++ ) {
	  buff_p_Yl[ i ] = i;
	}
	for ( i=0; i < nb_alg; i++ ) {
	  buff_p_bl[ i ] = i;
	}
    //
    // Compute QRP of panel AB1 = [ A11; A21 ].
    // Apply same permutations to A01 and Y1, and build T1_T.
    //

    if ( last_iter == 0 ) {

	  // read out block of matrix [ A12; A22; A32 ], applying the permutation as we read
	  for ( i=0; i < min( nb_alg, n_A - j ); i++ ) {
		
		fseek( A_fp, ( 0 + ( j + buff_p_Yl[ i ] ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
		err_check = fread( & buff_A_mid[ 0 + i * ldim_A_mid ], 
						   sizeof( double ), m_A_mid, A_fp );			   
		if ( err_check != m_A_mid ) {
		  printf( "Error! read of block [A12; A22; A32] failed!\n Number of entries read: %d; number of entries attempted: %d \n", (int) err_check, m_A_mid );
		  return 1;
		}
	  }
    }
	else {
	  fseek( A_fp, ( 0 + ( j ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
	  err_check = fread( buff_A_mid, sizeof( double ), m_A_mid * n_A_mid, A_fp );			   
	  if ( err_check != m_A_mid * n_A_mid ) {
		printf( "Error! read of block [A12; A22; A32] failed!\n Number of entries read: %d; number of entries attempted: %d \n", (int) err_check, m_A_mid * n_A_mid );
		return 1;
	  }
	}

	  // compute QRP; apply permutations to A12 
	  QRP_WY( panel_pivoting, -1,
        m_A_mid - j, n_A_mid, & buff_A_mid[ j + 0 * ldim_A_mid ], ldim_A_mid, buff_p_bl, buff_tl,
        1, 1, buff_pl, 1,
        1, j, buff_A_mid, ldim_A_mid,
        1, buff_Wl, ldim_W );

    //
    // Update the rest of the matrix.
    //
    if ( ( j + b ) < n_A ) {

	  // read out the old [A12; A22; A32] so we can perform the permutations as we update 
	  // the rest of the matrix 
	  fseek( A_fp, ( 0 + ( j ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
	  err_check = fread( buff_A_mid_old, sizeof( double ), m_A_mid_old * n_A_mid_old, A_fp );		
	  if ( err_check != m_A_mid_old * n_A_mid_old ) {
		printf( "Error! read of old block [A12; A22; A32] failed!\n" );
		return 1;
	  }

      // Apply the Householder transforms associated with [ A22; A32 ] 
      // and T1_T to [ A23; A33 ]:
      //   / A23 \ := QB1' / A23 \
      //   \ A33 /         \ A33 /
      // where QB1 is formed from AB1 and T1_T.
	  //
	  // also downdate Y3 with
	  // Y3 = Y3 - (G*Q(:,I2))*A23

	  // read in first block of cols of A to update
      n_work_current = min( num_cols_read, n_A - j - b );

	  fseek( A_fp, ( 0 + ( j + b ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
	  err_check = fread( buff_work_current, sizeof( double ), 
						 m_work * n_work_current, A_fp );			
	  if ( err_check != m_work * n_work_current ) {
		printf( "Error! read of block for A update failed!\n" );
		return 1;
	  }

	  // apply pivots to the block as necessary
	  for ( k=0; k < n_work_current; k++ ) {
		if ( buff_p_Yl[ b + k ] != b+k ) {
		  dlacpy_( "All", & m_work, & i_one, 
				   & buff_A_mid_old[ 0 + ( buff_p_Yl[ b + k ] ) * ldim_A_mid_old ], 
				   & ldim_A_mid_old,
				   & buff_work_current[ 0 + k * ldim_work ], & ldim_work );
		}
	  }

	  // do update
	  for ( i=j + b; i < n_A; i+=num_cols_read ) {

        n_work_current = min( num_cols_read, n_A - i );
        n_work_next = min( num_cols_read, n_A - i - num_cols_read );

        # pragma omp parallel sections
		{
	      # pragma omp section
		  {
		    if ( i + num_cols_read < n_A ) {
			  // extract one block of cols to update
			  fseek( A_fp, ( 0 + ( i + num_cols_read ) * ( ( long long int ) ldim_A ) ) * sizeof( double ), SEEK_SET );
			  err_check = fread( buff_work_next, sizeof( double ), 
								 m_work * n_work_next, A_fp );			
			  if ( err_check != m_work * n_work_next ) {
				printf( "Error! read of block for A update failed!\n" );
			  }

			  // apply pivots to the block as necessary
			  for ( k=0; k < n_work_next; k++ ) {
				if ( buff_p_Yl[ i + num_cols_read + k - j ] != i+num_cols_read+k-j ) {
				  dlacpy_( "All", & m_work, & i_one, 
						   & buff_A_mid_old[ 0 + ( buff_p_Yl[ i + num_cols_read + k - j ] ) * ldim_A_mid_old ], 
						   & ldim_A_mid_old,
						   & buff_work_next[ 0 + k * ldim_work ], & ldim_work );
				}
			  }
			}
		  } // end pragma omp section

		  # pragma omp section
		  {
			// apply update
			for ( k=0; k < n_work_current; k += nb_alg ) {
			  
			  int num_cols_updt = min( nb_alg, n_work_current - k );

			  Apply_Q_WY_lh( m_A_mid - j, n_A_mid, & buff_A_mid[ j + 0 * ldim_A_mid ], ldim_A_mid,
							  b, b, buff_Wl, ldim_W,
							  m_work - j, num_cols_updt, & buff_work_current[ j + k * ldim_work ], ldim_work );
			}
		  } // end pragma omp section
		
		} // end pragma omp parallel sections



		// write results back out
	    fseek( A_fp, 0 + ( i ) * ( ( long long int ) ldim_A  ) * sizeof( double ), SEEK_SET );
	    fwrite( buff_work_current, sizeof( double ), m_work * n_work_current, A_fp );			

		// copy over buff_work_next to buff_work_current for next iteration, if necessary
		if ( i + num_cols_read < n_A ) {
		// TODO: change to dlacpy
		  for ( k=0; k < m_work * n_work_next; k++ ) {
		    buff_work_current[ k ] = buff_work_next[ k ];
		  }
		}

	  }

    }

    /*
    Downdate_G( 
		  b, b, & buff_A_mid[ j + 0 * ldim_A_mid ], ldim_A_mid,
		  m_A_mid - j - b, n_A_mid, & buff_A_mid[ (j+b) + 0 * ldim_A1112 ], ldim_A_mid,
		  b, b, buff_Wl, ldim_W,
		  m_G, b, buff_G1, ldim_G,
		  m_G, max( 0, n_G - j - b ), buff_G2, ldim_G );
	*/

	// write out results to [A12; A22; A32]
	fseek( A_fp, 0 + ( j ) * ( ( long long int ) ldim_A  ) * sizeof( double ), SEEK_SET );
	fwrite( buff_A_mid, sizeof( double ), m_A_mid * n_A_mid, A_fp );			

  }

  // Remove auxiliary objects.
  fclose( A_fp );

  free( buff_G );
  free( buff_Y );
  free( buff_Yc );
  free( buff_W );
  free( buff_A_mid );
  free( buff_A_mid_old );
  free( buff_A1112 );
  free( buff_A21 );
  free( buff_work_current );
  free( buff_work_next );
  free( buff_p_Y );
  free( buff_p_bl );

  return 0;
}


// ============================================================================
static int Mult_BA_A_out( int m, int n, int k,
				FILE * A_fp, int ldim_A, double * B_p, int ldim_B,
				double * C_p, int ldim_C,
				int bl_size, int num_cols_read ) {
  // Computes C <-- B*A when matrix A is stored out of core,
  // and B and C can be stored in core
  // NOTE: the file position of the stream must be set correctly before entry!
  // TODO: this function is only called once in the main function and only works
  //       for that specific instance; change the description, name, and structure
  //       of this function to reflect that

  // bl_size is the number of cols of A that can be stored in RAM at a time

  // declare aux vars
  double * A_bl_p; // stores a block of cols of A
  int num_cols_block, b;
  size_t check;

  double d_one = 1.0, d_zero = 0.0;
  int i,j;

  // some initializations
  A_bl_p = ( double * ) malloc( k * num_cols_read * sizeof( double ) );

  // do multiplication one block at a time
  for ( i=0; i < n; i+= num_cols_read ) {
    
	num_cols_block = min( num_cols_read, n - i );
	
	// read a block of A into memory
	check = fread( A_bl_p, sizeof( double ), k * num_cols_block, A_fp ); 
	if ( ( int ) check != k * num_cols_block ) {
	  printf( "Warning! read failed in Mult_BA_A_out. check = %d \n", (int)check );  
	  return 1;
	}

    for ( j=0; j < num_cols_block; j+= bl_size ) {

      b = min( bl_size, num_cols_block - j );

	  // do multiplication; gives one block of cols of C
	  dgemm( "No transpose", "No tranpose", 
			  & m, & b, & k,
			  & d_one, B_p, & ldim_B, & A_bl_p[ 0 + j * k ], & k,
			  & d_zero, & C_p[ 0 + ( i + j ) * ldim_C ], & ldim_C );
    }

  }

  // free memory
  free( A_bl_p );

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
               int m_A12, int n_A12, double * buff_A12, int ldim_A12,
               int m_Y2, int n_Y2, double * buff_Y2, int ldim_Y2,
			   int m_Y_aux, int n_Y_aux, double * buff_Y_aux, int ldim_Y_aux) {
//
// It downdates matrix Y.
// Only Y2 of Y is updated.
//
// Y2 = Y2 - ( G1 - ( G1*U11 + G2*U21 ) * T11 * U11' ) * R12.
//    = Y2 - Y_updt_aux * R12
//
  int    i, j;
  double d_one       = 1.0;
  double d_minus_one = -1.0;

  // Y2 = Y2 - B * R12.
  //// FLA_Gemm( FLA_NO_TRANSPOSE, FLA_NO_TRANSPOSE,
  ////           FLA_MINUS_ONE, B, A12, FLA_ONE, Y2 );
  dgemm( "No transpose", "No transpose", & m_Y2, & n_Y2, & m_A12,
          & d_minus_one, buff_Y_aux, & ldim_Y_aux, buff_A12, & ldim_A12,
          & d_one, buff_Y2, & ldim_Y2 );

  return 0;
}

// ============================================================================
static int Form_Y_aux( 
               int m_U11, int n_U11, double * buff_U11, int ldim_U11,
               int m_U21, int n_U21, double * buff_U21, int ldim_U21,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_G1, int n_G1, double * buff_G1, int ldim_G1,
               int m_G2, int n_G2, double * buff_G2, int ldim_G2,
			   int m_Y_aux, int n_Y_aux, double * buff_Y_aux, int ldim_Y_aux ) {
//
// It forms the auxiliary matrix in the Y downdate:
// Y_aux = ( G1 - ( G1 * U11 + G2 * U21 ) * T11 * U11' )
//
  int    i, j;
  double d_one       = 1.0;
  double d_minus_one = -1.0;

  // Y_aux = G1.
  //// FLA_Copy( G1, Y_aux );
  dlacpy_( "All", & m_G1, & n_G1, buff_G1, & ldim_G1,
                                  buff_Y_aux, & ldim_Y_aux );

  // Y_aux = Y_aux * U11.
  //// FLA_Trmm( FLA_RIGHT, FLA_LOWER_TRIANGULAR,
  ////           FLA_NO_TRANSPOSE, FLA_UNIT_DIAG,
  ////           FLA_ONE, U11, Y_aux );
  dtrmm( "Right", "Lower", "No transpose", "Unit", & m_Y_aux, & n_Y_aux,
          & d_one, buff_U11, & ldim_U11, buff_Y_aux, & ldim_Y_aux );

  // Y_aux = Y_aux + G2 * U21.
  //// FLA_Gemm( FLA_NO_TRANSPOSE, FLA_NO_TRANSPOSE,
  ////           FLA_ONE, G2, U21, FLA_ONE, Y_aux );
  dgemm( "No transpose", "No tranpose", & m_Y_aux, & n_Y_aux, & m_U21,
          & d_one, buff_G2, & ldim_G2, buff_U21, & ldim_U21,
          & d_one, buff_Y_aux, & ldim_Y_aux );

  // Y_aux = Y_aux * T11.
  //// FLA_Trsm( FLA_RIGHT, FLA_UPPER_TRIANGULAR,
  ////           FLA_NO_TRANSPOSE, FLA_NONUNIT_DIAG,
  ////           FLA_ONE, T, Y_aux );
  //// dtrsm_( "Right", "Upper", "No transpose", "Non-unit", & m_Y_aux, & n_Y_aux,
  ////         & d_one, buff_T, & ldim_T, buff_Y_aux, & ldim_Y_aux );
  // Used dtrmm instead of dtrsm because of using compact WY instead of UT.
  dtrmm( "Right", "Upper", "No transpose", "Non-unit", & m_Y_aux, & n_Y_aux,
          & d_one, buff_T, & ldim_T, buff_Y_aux, & ldim_Y_aux );

  // Y_aux = - Y_aux * U11^H.
  //// FLA_Trmm( FLA_RIGHT, FLA_LOWER_TRIANGULAR,
  ////           FLA_CONJ_TRANSPOSE, FLA_UNIT_DIAG,
  ////           FLA_MINUS_ONE, U11, Y_aux );
  dtrmm( "Right", "Lower", "Conj_tranpose", "Unit", & m_Y_aux, & n_Y_aux,
          & d_minus_one, buff_U11, & ldim_U11, buff_Y_aux, & ldim_Y_aux );

  // Y_aux = G1 + Y_aux.
  //// FLA_Axpy( FLA_ONE, G1, Y_aux );
  for( j = 0; j < n_Y_aux; j++ ) {
    for( i = 0; i < m_Y_aux; i++ ) {
      buff_Y_aux[ i + j * ldim_Y_aux ] += buff_G1[ i + j * ldim_G1 ];
    }
  }

  return 0;
}

// ============================================================================
static int Downdate_G( 
               int m_U11, int n_U11, double * buff_U11, int ldim_U11,
               int m_U21, int n_U21, double * buff_U21, int ldim_U21,
               int m_T, int n_T, double * buff_T, int ldim_T,
               int m_G1, int n_G1, double * buff_G1, int ldim_G1,
               int m_G2, int n_G2, double * buff_G2, int ldim_G2 ) {
  // Updates matrix G
  // Only G1 and G2 of G are updated.

  //
  // GR = GR * Q
  //
  NoFLA_Apply_Q_WY_rnfc_blk_var4( 
          m_U11 + m_U21, n_U11, buff_U11, ldim_U11,
          m_T, n_T, buff_T, ldim_T,
          m_G1, n_G1 + n_G2, buff_G1, ldim_G1 );

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
static int Apply_pivot_ooc( int m_A, int n_A, FILE * A_fp, int ldim_A,
				int * buff_p ) {
  // applies a permutation vector to the matrix stored in the file
  // pointed to by A_fp



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
// pivot vector B and matrix C are optionally pivoted.
//
  int     ival, i_one = 1;
  double  * ptr_g1, * ptr_g2, * ptr_c1, * ptr_c2;
  int * ptr_b1, * ptr_b2;
  int btemp;

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
