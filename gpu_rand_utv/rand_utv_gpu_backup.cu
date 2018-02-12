#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusolverDn.h>

#include <magma.h>
#include <magma_lapack.h>

// =======================================================================
// Definition of macros

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) > (b) ? (b) : (a) )
#define dabs( a, b ) ( (a) >= 0.0 ? (a) : -(a) )

// =======================================================================
// Compilation declarations

//#define PROFILE

// =======================================================================
// Definition of global variables

static int gpu_thread_size = 256; // TODO: temporary value; determine how to optimize
						 // this gives the number of threads per block for
						 // kernel calls

clock_t T1, T2;
double test_time = 0.0; // TODO: delete once we're finished profiling

// =======================================================================
// TODO: delete this function once we're finished building
static void gpu_print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A ) {
  int  i, j;

  double * A_pc;

  A_pc = ( double * ) malloc( m_A * n_A * sizeof( double ) );

  cudaMemcpy( A_pc, buff_A, m_A * n_A * sizeof( double ),
			cudaMemcpyDeviceToHost );

  cudaDeviceSynchronize();
  printf( "%s = [\n", name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%.16e ", A_pc[ i + j * ldim_A ] );
    }
    printf( "\n" );
  }
  printf( "];\n" );

  free( A_pc );
}

static void print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A ) {
  int  i, j;

  printf( "%s = [\n", name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%.16e ", buff_A[ i + j * ldim_A ] );
    }
    printf( "\n" );
  }
  printf( "];\n" );

  free( buff_A );
}
// ========================================================================
// Declaration of local prototypes

static void Allocate_work_array( cusolverDnHandle_t handle,
				double ** work_pg_p,
				int m_A, int n_A, double * A_pg, int ldim_A,
				int bl_size );

static void Set_to_identity( int m_A, int n_A, double * A_pc, int ldim_A );

static void Normal_random_matrix( int m_A, int n_A,
				double * A_pg, int ldim_A,
				curandGenerator_t rand_gen );

static void local_magqr_nopiv( int m_A, int n_A, 
				double * A_pg, int ldim_A,
				double * tau_h, magma_int_t * magInfo );

static void gpu_orth( int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_h, double * T_d,
				magma_int_t * magInfo );

static void local_magormqr( magma_side_t mag_side, magma_trans_t mag_trans,
				int m_C, int n_C, double * C_pg, int ldim_C,
				int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_h,
				double * work_h, double * T_d,
				magma_int_t * magInfo ); 

// TODO: if magma is faster, get rid of this old cusolver dgeqrf and 
// dormqr wrapper in the code
static void gpu_dgeqrf( cusolverDnHandle_t handle,
				int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_pg, 
				double * work_pg, int * devInfo );

__global__
static void Make_upper_tri_kern( int m_A, int n_A,
				int ldim_A, double * A_pg );

static void Make_upper_tri( int m_A, int n_A,
				int ldim_A, double * A_pg );

__global__
static void Set_ss_diag_mat_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * ss_pg );

static void Set_ss_diag_mat( int m_A, int n_A, double * A_pg, int ldim_A,
				double * ss_pg );

static void local_magsvd( int m_A, int n_A, double * A_pg, int ldim_A,
                double * ss_pg,
                double * U_pg, int ldim_U,
                double * Vt_pg, int ldim_Vt,
				double * work_h, int * iwork_h );

static void local_left_svecs( int m_A, int n_A, double * A_pg, int ldim_A,
				double * work_h, int * iwork_h );

static void gpu_dgesvd( cusolverDnHandle_t handle,
				int m_A, int n_A, double * A_pg, int ldim_A,
				double * ss_pg,
				double * U_pg, int ldim_U,
				double * Vt_pg, int ldim_Vt,
				double * work_pg, int * devInfo );

static void gpu_dgemm( cublasHandle_t handle,
				char opA, char opB, double alpha, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				int m_B, int n_B, double * B_pg, int ldim_B,
				double beta,
				int m_C, int n_C, double * C_pg, int ldim_C );

static void dlarft_gpu( cublasHandle_t handle, int n, int k, 
				double * V_pg, int ldim_V,
				double * tau_pg,
				double * T_pg, int ldim_T,
				double * TVt_pg, int ldim_TVt_pg,
				double * work_pg );

static void magma_dlarft( cublasHandle_t handle, int n, int k, 
				double * V_pg, int ldim_V,
				double * tau_h,
				double * T_pg, int ldim_T,
				double * TVt_pg, int ldim_TVt );

__global__
static void replace_diag_ones_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * work_pg ); 

static void replace_diag_ones( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * work_pg );

__global__
static void replace_diag_vec_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * vec_pg );

static void replace_diag_vec( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * vec_pg );

__global__
static void check_tau_zeroes_kern( int m_T, int n_T,
				double * T_pg, int ldim_T,
				double * tau_pg );

static void check_tau_zeroes( int m_T, int n_T,
				double * T_pg, int ldim_T,
				double * tau_pg );

static void my_dormqr_gpu( cublasHandle_t handle,
				char side, char trans, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				int m_B, int n_B, double * B_pg, int ldim_B,
				double * TVt_pg, int ldim_TVt,
				double * work_pg ); 
__global__
static void mm_A_minus_B_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * B_pg, int ldim_B );

static void mm_A_minus_B( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * B_pg, int ldim_B );

__global__
static void mm_A_plus_B_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * B_pg, int ldim_B );

static void mm_A_plus_B( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * B_pg, int ldim_B );

__global__
static void set_vc_zero_kern( int lvec, double * vec_pg );

static void set_vc_zero( int lvec, double * vec_pg );

__global__
static void set_vc_one_kern( int lvec, double * vec_pg );

static void set_vc_one( int lvec, double * vec_pg );

// ========================================================================

// Main function
int rand_utv_gpu( 
		int m_A, int n_A, double * A_pc, int ldim_A,
		int build_U, int m_U, int n_U, double * U_pc, int ldim_U,
		int build_V, int m_V, int n_V, double * V_pc, int ldim_V,
		int bl_size, int pp, int q_iter, int ON ) {

// randUTV: It computes the (rank-revealing) UTV factorization of matrix A.
//
// Matrices A, U, V must be stored in column-order
//
// we assume m_A >= n_A TODO: add check to make sure this is true, decide what to do if not 
//
// Arguments:
// ----------
// m_A:      Number of rows of matrix A.
// n_A:      Number of columns of matrix A.
// A_pc:   Address of data in matrix A in cpu. Matrix to be factorized.
// ldim_A:   Leading dimension of matrix A.
// build_U:  If build_U==1, matrix U is built.
// m_U:      Number of rows of matrix U.
// n_U:      Number of columns of matrix U.
// U_pc:   Address of data in matrix U in cpu.
// ldim_U:   Leading dimension of matrix U.
// build_V:  If build_V==1, matrix V is built.
// m_V:      Number of rows of matrix V.
// n_V:      Number of columns of matrix V.
// V_pc:   Address of data in matrix V in cpu.
// ldim_V:   Leading dimension of matrix V.
// bl_size:   Block size. Usual values for nb_alg are 32, 64, etc.
// pp:       Oversampling size. Usual values for pp are 5, 10, etc.
// q_iter:   Number of "power" iterations. Usual values are 2.
// ON:		If ON==1, orthonormalization is used in power iteration


  // Declaration of variables
  double d_one = 1.0;
  double d_zero = 0.0;
  char t = 'T', n = 'N', l = 'L', r = 'R';
  magma_side_t mag_side_l = MagmaLeft, mag_side_r = MagmaRight;
  magma_trans_t mag_trans_t = MagmaTrans, mag_trans_n = MagmaNoTrans;
  int i, j;
  int mn_A, mx_A,  num_spl;
  int m_G, n_G, ldim_G, m_G_loc, n_G_loc,
	  m_Y, n_Y, ldim_Y, m_Y_loc, n_Y_loc;
  int m_A_loc, n_A_loc; // this block is comprised of sections 22,23,32,33
  int m_A_right, n_A_right; // this block is comprised of sections
							// 12, 13, 22, 23, 32, 33
  int m_A_bl, n_A_bl; // this block is comprised of 22 and 23
  int m_A_BR, n_A_BR; // this block is comprised of 32 and 33
  int m_A_22, n_A_22; // this is the 22 block of the matrix
  int m_A_12, n_A_12; // this is the 12 block of the matrix
  int m_A_23, n_A_23; // this is the 23 block of the matrix
  int m_V_right, n_V_right;
  int m_U_right, n_U_right;
  int m_V_mid, n_V_mid; // this is the middle block of cols affected by
						// the small SVD
  int m_U_mid, n_U_mid; // this is the middle block of cols affected by
						// the small SVD
  int m_Vt_svd, n_Vt_svd, ldim_Vt_svd;
  int m_U_svd, n_U_svd, ldim_U_svd;
  int m_T, n_T, ldim_T; // holds the T matrix in a UT representation of
						// HH matrices
  int ldim_TVt;
  double * A_pg, * U_pg, * V_pg; // pointers to matrix arrays in gpu
  double * G_pg, * G_loc_pg, * A_loc_pg, * A_right_pg, * A_bl_pg, * A_BR_pg,
		 * A_22_pg, * A_12_pg, * A_23_pg, * Y_pg, * Y_loc_pg;
  double * V_right_pg, * U_right_pg;
  double * V_mid_pg, * U_mid_pg;
  double * Vt_svd_pg, * U_svd_pg;
  double * Tmp_pg; // this array will be a temporary holding place for
				   // some outputs of dgemm; NOTE: the leading dimension
				   // for this matrix should always match the number of 
				   // ROWS of the matrix for which it is standing in
  double * TVt_pg; // this matrix holds the product T*V' for use in applying HH transforms
				   // TODO: change name of TVt to something less confusing (TWt?)
  double * tau_pg;
  double * T_pg;
  double * ss_pg; 
 
  curandGenerator_t rand_gen;
  unsigned long long rand_seed = 7;

  // create pseudo-random number generator and set seed
  curandCreateGenerator( & rand_gen, CURAND_RNG_PSEUDO_DEFAULT );
  curandSetPseudoRandomGeneratorSeed( rand_gen, rand_seed );
  
  // create a handles for cublas and cusolver contexts
  cublasHandle_t cublasH;
  cublasCreate( & cublasH );

  cusolverDnHandle_t cusolverH = NULL; 
  cusolverDnCreate( & cusolverH );

  // initialize magma
  magma_init();

  // work array used by cusolver functions
  double * work_pg = NULL;

  int * devInfo = NULL; // another var for cusolver functions

  // work arrays for magma functions
  double * work_h;
  int * iwork_h;
  int lwork_mx, work_nb_svd, work_nb_qr; 
				// for determining size of work array
  magma_int_t * magInfo;
  double * T_d; // T matrix for applying HH matrices
  double * tau_h; // tau vector for applying HH matrices

#ifdef PROFILE
  clock_t t1, t2;
  double tt_spl,
		 tt_qr1_fact, tt_qr1_updt_a, tt_qr1_updt_v,
		 tt_qr2_fact, tt_qr2_updt_a, tt_qr2_updt_u,
		 tt_svd_fact, tt_svd_updt_a, tt_svd_updt_uv;
  
  timespec time1, time2; // vars for alternate timing method 
  uint64_t diff;
#endif

 

  // check matrix dimensions
    if( m_U != n_U ) {
	  fprintf( stderr, "rand_utv_gpu: Matrix U should be square.\n" ); 
	  exit( -1 );
	}
    if( m_V != n_V ) {
	  fprintf( stderr, "rand_utv_gpu: Matrix V should be square.\n" ); 
	  exit( -1 );
	}
    if( m_U != m_A ) {
	  fprintf( stderr, "rand_utv_gpu: Dims. of U and A do not match.\n");
	  exit( -1 );
	}
	if( n_A != m_V ) {
	  fprintf( stderr, "rand_utv_gpu: Dims. of A and V do not match.\n");
      exit( -1 );
    }

#ifdef PROFILE
  tt_spl = 0.0;
  tt_qr1_fact = 0.0;
  tt_qr1_updt_a = 0.0;
  tt_qr1_updt_v = 0.0;
  tt_qr2_fact = 0.0;
  tt_qr2_updt_a = 0.0;
  tt_qr2_updt_u = 0.0;
  tt_svd_fact = 0.0;
  tt_svd_updt_a = 0.0;
  tt_svd_updt_uv = 0.0;
#endif

  // initialize auxiliary variables
  mn_A = min( m_A, n_A );
  m_A_loc = m_A; n_A_loc = n_A;
  m_G = m_A; n_G = bl_size + pp; ldim_G = m_A;
  m_Y = n_A; n_Y = bl_size + pp; ldim_Y = n_A;
 
  m_Vt_svd = bl_size; n_Vt_svd = bl_size; ldim_Vt_svd = bl_size;
  m_U_svd = bl_size; n_U_svd = bl_size; ldim_U_svd = bl_size;

  m_T = bl_size; n_T = bl_size; ldim_T = bl_size;
  ldim_TVt = bl_size;

  // initialize auxiliary objects
  cudaError_t cudaStat = cudaSuccess; 
  cudaStat = cudaMalloc( & A_pg, m_A * n_A * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & U_pg, m_U * n_U * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & V_pg, m_V * n_V * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & G_pg, m_G * n_G * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & Y_pg, m_Y * n_Y * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  cudaStat = cudaMalloc( & Vt_svd_pg, bl_size * bl_size * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & U_svd_pg, bl_size * bl_size * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  cudaStat = cudaMalloc( & tau_pg, bl_size * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & T_pg, bl_size * bl_size * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & TVt_pg, m_A * bl_size * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & ss_pg, ( bl_size + pp ) * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  cudaStat = cudaMalloc( & Tmp_pg, m_A * bl_size * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  Allocate_work_array( cusolverH, 
				& work_pg, 
				m_A, n_A, A_pg, ldim_A,
				bl_size + pp );

  cudaMalloc( ( void ** ) & devInfo, sizeof( int ) );
  
  //TODO: get rid of compilation warnings

  // determine max size of work array for magma svd calcs,
  // allocate memory for the arrays
  mx_A = max( m_A, n_A ); mn_A = min( m_A, n_A );
  work_nb_svd = magma_get_dgesvd_nb( bl_size, bl_size );
  work_nb_qr = magma_get_dgeqrf_nb( m_A, bl_size + pp );
  lwork_mx = max((m_A-(bl_size+pp)+work_nb_qr)*(n_A+work_nb_qr)+n_A*work_nb_qr,(n_A-(bl_size+pp)+work_nb_qr)*(m_A+work_nb_qr)+m_A*work_nb_qr);
  lwork_mx = max( lwork_mx,3*(bl_size+pp) + max(3*(bl_size+pp)*(bl_size+pp)+4*(bl_size+pp),2*(bl_size+pp)*work_nb_svd) );
  lwork_mx = max( lwork_mx, (bl_size + pp)*(bl_size + pp)+3*(bl_size+pp)+max(3*(bl_size+pp)*(bl_size+pp)+4*(bl_size+pp),2*(bl_size+pp)*magma_get_sgesvd_nb(m_A,bl_size+pp)) );		
		//TODO: YOU BETTER MOVE THIS CRAP TO AN AUX FUNCTION

  magma_dmalloc_pinned( & work_h, lwork_mx );
  magma_imalloc_pinned( & iwork_h, 8 * ( bl_size + pp ) );

  magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );

  // determine max size of work arrays for magma qr calcs
  // and allocate memory for the arrays
  cudaMalloc( ( void ** ) & T_d, 
			(2*mn_A+ceil(n_A/32.0)*32)*magma_get_dgeqrf_nb( m_A, (bl_size+pp) ) * sizeof( double ));
  tau_h = ( double * ) malloc( ( bl_size + pp ) * sizeof( double ) );

  // copy A to gpu
  cudaMemcpy( A_pg, A_pc, m_A * n_A * sizeof( double ),
				cudaMemcpyHostToDevice );

  // initialize U and V to identity, transfer to gpu
  Set_to_identity( m_U, n_U, U_pc, ldim_U );
  cudaMemcpy( U_pg, U_pc, m_U * n_U * sizeof( double ),
			cudaMemcpyHostToDevice );
  
  Set_to_identity( m_V, n_V, V_pc, ldim_V );
  cudaMemcpy( V_pg, V_pc, m_V * n_V * sizeof( double ),
			cudaMemcpyHostToDevice );


  // Main loop
  for ( i=0; i < mn_A; i += bl_size ) {
    // some initializations for every iteration
	num_spl = min( bl_size, n_A - i );
	
	m_A_loc = m_A - i; 
	n_A_loc = n_A - i;
	m_G_loc = m_A - i;
	n_G_loc = num_spl + pp;
    m_Y_loc = n_A - i;
	n_Y_loc = num_spl + pp;
    m_A_right = m_A;
	n_A_right = n_A - i;
	m_V_right = n_A;
	n_V_right = n_A - i;
	m_A_bl = m_A - i;
	n_A_bl = num_spl;
	m_A_BR = m_A - i;
	n_A_BR = n_A - i - num_spl;
	m_U_right = m_U;
	n_U_right = n_U - i;
	m_A_22 = num_spl;
	n_A_22 = num_spl;
	m_Vt_svd = num_spl;
	n_Vt_svd = num_spl;
	m_U_svd = num_spl;
	n_U_svd = num_spl;
	m_A_12 = i;
	n_A_12 = num_spl;
	m_A_23 = num_spl;
	n_A_23 = n_A - i - num_spl;
	m_U_mid = m_U;
	n_U_mid = num_spl;
	m_V_mid = m_V;
	n_V_mid = num_spl;
	m_T = num_spl;
	n_T = num_spl;
    
	A_loc_pg = & A_pg[ i + i * ldim_A ];
	A_right_pg = & A_pg[ 0 + i * ldim_A ];
	G_loc_pg = & G_pg[ i + 0 * ldim_G ];
    Y_loc_pg = & Y_pg[ i + 0 * ldim_Y ]; 
    V_right_pg = & V_pg[ 0 + i * ldim_V ];
	A_bl_pg = & A_pg[ i + i * ldim_A ];
	A_BR_pg = & A_pg[ i + ( i + num_spl ) * ldim_A ];
	U_right_pg = & U_pg[ 0 + i * ldim_U ];
	A_22_pg = & A_pg[ i + i * ldim_A ];
    A_12_pg = & A_pg[ 0 + i * ldim_A ];
	A_23_pg = & A_pg[ i + ( i + num_spl ) * ldim_A ];
	U_mid_pg = & U_pg[ 0 + i * ldim_U ];
	V_mid_pg = & V_pg[ 0 + i * ldim_V ];

	// Compute the "sampling" matrix Y
	// Aloc = T([J2,I3],[J2,I3]);
	// Y = Aloc' * randn(m-(i-1)*b,b+p);
#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime( CLOCK_MONOTONIC, & time1 );
#endif
      // Create random matrix for sampling
      Normal_random_matrix( m_G, n_G, G_pg, ldim_G,
					rand_gen );
		// TODO: this currently fills the ENTIRE array G each loop;
		// unecessary, but does it matter? decide

      // carry out "sampling" multiplication
	  gpu_dgemm( cublasH,
					t, n, d_one,
					m_A_loc, n_A_loc, A_loc_pg, ldim_A,
					m_G_loc, n_G_loc, G_loc_pg, ldim_G,
					d_zero,
					m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y );
	
	// perform "power iteration" if requested
	// for i_iter = 1:q_iter:
	//   Y = Aloc' * (Aloc * Y);
	// end
    for( j=0; j<q_iter; j++ ) {
	 
	  if ( ON == 1 ) {
		gpu_orth( m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y,
				  tau_h, T_d,
				  magInfo );
	  }

	  // reuse G_loc for storage; G <-- A*Y
	  gpu_dgemm( cublasH, 
					n, n, d_one,
					m_A_loc, n_A_loc, A_loc_pg, ldim_A,
					m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y,
					d_zero, 
					m_G_loc, n_G_loc, G_loc_pg, ldim_G );
	  
	  // complete iteration; Y <-- A'*G
	  gpu_dgemm( cublasH, 
					t, n, d_one,
					m_A_loc, n_A_loc, A_loc_pg, ldim_A,
					m_G_loc, n_G_loc, G_loc_pg, ldim_G,
					d_zero,
					m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y );
	}

	// if oversampling is done, extract the basis basis of bl_size vectors
	if ( pp > 0 ) {
	  // compute left singular vectors of sampling matrix; use first bl_size 
	  // vectors as new sampling matrix to compute left transformation
	  local_left_svecs( m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y,
				work_h, iwork_h );
	  
	}

// TODO: clean up the timing code
#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &time2);
	diff = (1000000000L) * (time2.tv_sec - time1.tv_sec) + time2.tv_nsec - time1.tv_nsec;
	tt_spl += (double) diff / (1000000000L);
	clock_gettime(CLOCK_MONOTONIC, &time1);
#endif
  
    // Construct the local transform to be applied "from the right".
    // if (p > 0)
    //   [~,~,Jtmp] = qr(Y,0);
    //   [Vloc,~,~] = qr(Y(:,Jtmp(1:b)));
    // else
    //   [Vloc,~]   = LOCAL_nonpiv_QR(Y,b);
    // end

	local_magqr_nopiv( m_Y_loc, n_Y_loc-pp, 
				Y_loc_pg, ldim_Y,
				tau_h, magInfo ); 

	//gpu_dgeqrf( cusolverH, 
	//				m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y, 
	//				tau_pg, 
	//				work_pg, devInfo );
    
	// construct "TU'" matrix for UT representation of HH matrix
	magma_dlarft( cublasH, m_Y_loc, n_Y_loc-pp, 
				Y_loc_pg, ldim_Y,
				tau_h,
				T_pg, ldim_T,
				TVt_pg, ldim_TVt );
    //dlarft_gpu( cublasH,
	//				m_Y_loc, n_Y_loc, 
	//				Y_loc_pg, ldim_Y,
	//				tau_pg,
	//				T_pg, ldim_T,
	//				TVt_pg, ldim_TVt,
	//				work_pg );




	// TODO: remove capability for oversampling; it's unecessary
#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &time2);
	diff = (1000000000L) * (time2.tv_sec - time1.tv_sec) + time2.tv_nsec - time1.tv_nsec;
	tt_qr1_fact += (double) diff / (1000000000L);
	clock_gettime(CLOCK_MONOTONIC, &time1);
#endif
    // Apply the pivot matrix to rotate maximal mass into the "J2" column
	// T(:,[J2,J3]) = T(:[J2,J3])*Vloc;

    my_dormqr_gpu( cublasH,
					r, n, 
					m_Y_loc, n_Y_loc-pp, Y_loc_pg, ldim_Y,
					m_A_right, n_A_right, A_right_pg, ldim_A,
					TVt_pg, ldim_TVt,
					work_pg );

#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &time2);
	diff = (1000000000L) * (time2.tv_sec - time1.tv_sec) + time2.tv_nsec - time1.tv_nsec;
	tt_qr1_updt_a += (double) diff / (1000000000L);
	clock_gettime(CLOCK_MONOTONIC, &time1);
#endif

    // Update matrix V with transformations from the first QR.
	my_dormqr_gpu( cublasH,
					r,n,
					m_Y_loc, n_Y_loc-pp, Y_loc_pg, ldim_Y,
					m_V_right, n_V_right, V_right_pg, ldim_V,
					TVt_pg, ldim_TVt,
					work_pg );

#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime( CLOCK_MONOTONIC, & time2 );
	diff = (1000000000L) * ( time2.tv_sec - time1.tv_sec ) + 
			time2.tv_nsec - time1.tv_nsec;
	tt_qr1_updt_v += (double) diff / (1000000000L);
	clock_gettime( CLOCK_MONOTONIC, & time1 );
#endif
    
	// %%% Next determine the rotations to be applied "from the left".
    // [Uloc,Dloc]      = LOCAL_nonpiv_QR(T([J2,I3],J2));

	local_magqr_nopiv( m_A_bl, n_A_bl, 
				A_bl_pg, ldim_A,
				tau_h, magInfo );
	//gpu_dgeqrf( cusolverH, 
	//				m_A_bl, n_A_bl, A_bl_pg, ldim_A,
	//				tau_pg, 
	//				work_pg, devInfo );
	
	magma_dlarft( cublasH, m_A_bl, n_A_bl, 
				A_bl_pg, ldim_A,
				tau_h,
				T_pg, ldim_T,
				TVt_pg, ldim_TVt );
	
	//dlarft_gpu( cublasH,
	//				m_A_bl, n_A_bl, 
	//				A_bl_pg, ldim_A,
	//				tau_pg,
	//				T_pg, ldim_T,
	//				TVt_pg, ldim_TVt,
	//				work_pg );

#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime( CLOCK_MONOTONIC, & time2 );
	diff = (1000000000L) * ( time2.tv_sec - time1.tv_sec ) + 
			time2.tv_nsec - time1.tv_nsec;
	tt_qr2_fact += (double) diff / (1000000000L);
	clock_gettime( CLOCK_MONOTONIC, & time1 );
#endif

	// update rest of matrix A with transformations from the second QR
	my_dormqr_gpu( cublasH,
					l, t,
					m_A_bl, n_A_bl, A_bl_pg, ldim_A,
					m_A_BR, n_A_BR, A_BR_pg, ldim_A,
					TVt_pg, ldim_TVt,
					work_pg );

#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime( CLOCK_MONOTONIC, & time2 );
	diff = (1000000000L) * ( time2.tv_sec - time1.tv_sec ) + 
			time2.tv_nsec - time1.tv_nsec;
	tt_qr2_updt_a += (double) diff / (1000000000L);
	clock_gettime( CLOCK_MONOTONIC, & time1 );
#endif

	// update matrix U with transformations from the second QR
	my_dormqr_gpu( cublasH,
					r, n,
					m_A_bl, n_A_bl, A_bl_pg, ldim_A,
					m_U_right, n_U_right, U_right_pg, ldim_U,
					TVt_pg, ldim_TVt,
					work_pg );

#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime( CLOCK_MONOTONIC, & time2 );
	diff = (1000000000L) * ( time2.tv_sec - time1.tv_sec ) + 
			time2.tv_nsec - time1.tv_nsec;
	tt_qr2_updt_u += (double) diff / (1000000000L);
	clock_gettime( CLOCK_MONOTONIC, & time1 );
#endif
	
	// get rid of HH reflectors to make A upper triangular
	Make_upper_tri( m_A_bl, n_A_bl, ldim_A, A_bl_pg );

    // Compute miniSVD, update A, update U, update V
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

	
	// compute SVD
	
	// TODO: remove old function gpu_dgesvd which uses cusolver from
	// if it turns out to be slow
	local_magsvd( m_A_22, n_A_22, A_22_pg, ldim_A,
					ss_pg, U_svd_pg, ldim_U_svd,
					Vt_svd_pg, ldim_Vt_svd,
					work_h, iwork_h );

#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime( CLOCK_MONOTONIC, & time2 );
	diff = (1000000000L) * ( time2.tv_sec - time1.tv_sec ) + 
			time2.tv_nsec - time1.tv_nsec;
	tt_svd_fact += (double) diff / (1000000000L);
	clock_gettime( CLOCK_MONOTONIC, & time1 );
#endif
	
	// update A
	gpu_dgemm( cublasH,
					n, t, d_one,
					m_A_12, n_A_12, A_12_pg, ldim_A,
					m_Vt_svd, n_Vt_svd, Vt_svd_pg, ldim_Vt_svd,
					d_zero,
					m_A_12, n_A_12, Tmp_pg, max( m_A_12, 1 ) );

	// copy from temporary buffer to A
	magma_dcopymatrix( m_A_12, n_A_12, Tmp_pg, m_A_12, A_12_pg, ldim_A );
	
	gpu_dgemm( cublasH,
					t, n, d_one,
					m_U_svd, n_U_svd, U_svd_pg, ldim_U_svd,
					m_A_23, n_A_23, A_23_pg, ldim_A,
					d_zero,
					m_A_23, n_A_23, Tmp_pg, m_A_23 );
	
	// copy from temporary buffer to A
	magma_dcopymatrix( m_A_23, n_A_23, Tmp_pg, m_A_23, A_23_pg, ldim_A );

#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime( CLOCK_MONOTONIC, & time2 );
	diff = (1000000000L) * ( time2.tv_sec - time1.tv_sec ) + 
			time2.tv_nsec - time1.tv_nsec;
	tt_svd_updt_a += (double) diff / (1000000000L);
	clock_gettime( CLOCK_MONOTONIC, & time1 );
#endif
    
	// update U
	gpu_dgemm( cublasH,
					n, n, d_one,
					m_U_mid, n_U_mid, U_mid_pg, ldim_U,
					m_U_svd, n_U_svd, U_svd_pg, ldim_U_svd,
					d_zero,
					m_U_mid, n_U_mid, Tmp_pg, m_U_mid );	

	// copy from temporary buffer to U
	magma_dcopymatrix( m_U_mid, n_U_mid, Tmp_pg, m_U_mid, U_mid_pg, ldim_U );

	// update V
	gpu_dgemm( cublasH,
					n, t, d_one,
					m_V_mid, n_V_mid, V_mid_pg, ldim_V,
					m_Vt_svd, n_Vt_svd, Vt_svd_pg, ldim_Vt_svd,
					d_zero,
					m_V_mid, n_V_mid, Tmp_pg, m_V_mid );
	
	// copy from temporary buffer to V
	magma_dcopymatrix( m_V_mid, n_V_mid, Tmp_pg, m_V_mid, V_mid_pg, ldim_V );

#ifdef PROFILE
	cudaDeviceSynchronize();
	clock_gettime( CLOCK_MONOTONIC, & time2 );
	diff = (1000000000L) * ( time2.tv_sec - time1.tv_sec ) + 
			time2.tv_nsec - time1.tv_nsec;
	tt_svd_updt_uv += (double) diff / (1000000000L);
#endif

	// end of main loop  
  }

  // the final, potentially abnormally-sized block is processed inside the
  // previous loop

  // transfer arrays from device to host
  cudaStat = cudaMemcpy( A_pc, A_pg, m_A * n_A * sizeof( double ),
				cudaMemcpyDeviceToHost );
  assert( cudaStat == cudaSuccess );

  cudaStat = cudaMemcpy( U_pc, U_pg, m_U * n_U * sizeof( double ),
			cudaMemcpyDeviceToHost );
  assert( cudaStat == cudaSuccess );
  
  cudaStat = cudaMemcpy( V_pc, V_pg, m_V * n_V * sizeof( double ),
			cudaMemcpyDeviceToHost );
  assert( cudaStat == cudaSuccess );
  
  // remove auxiliary objects
  cudaFree( A_pg );
  cudaFree( U_pg );
  cudaFree( V_pg );

  cudaFree( G_pg );
  cudaFree( Y_pg );
  cudaFree( Vt_svd_pg );
  cudaFree( U_svd_pg );
  cudaFree( tau_pg );
  cudaFree( T_pg );
  cudaFree( TVt_pg );
  cudaFree( ss_pg );

  magma_free_pinned( work_h );
  magma_free_pinned( iwork_h );

  free( magInfo );

  cudaFree( T_d );
  free( tau_h );

  // finalize magma
  magma_finalize();
  
  // destroy the handles for the blas and solver environments
  cublasDestroy( cublasH );
  cusolverDnDestroy( cusolverH );

  cudaFree( work_pg );
  cudaFree( devInfo );

#ifdef PROFILE
  printf( "%% tt_build_y:	%le \n", tt_spl );
  printf( "%% tt_qr1:	%le \n", tt_qr1_fact + tt_qr1_updt_a + 
								 tt_qr1_updt_v );
  printf( "%%	tt_qr1_fact:	%le \n", tt_qr1_fact );
  printf( "%%	tt_qr1_updt_a:	%le \n", tt_qr1_updt_a );
  printf( "%%	tt_qr1_updt_v:	%le \n", tt_qr1_updt_v );
  printf( "%% tt_qr2:	%le \n", tt_qr2_fact + tt_qr2_updt_a +
									 tt_qr2_updt_u );
  printf( "%%	tt_qr2_fact:	%le \n", tt_qr2_fact );
  printf( "%%	tt_qr2_updt_a:	%le \n", tt_qr2_updt_a );
  printf( "%%	tt_qr2_updt_u:	%le \n", tt_qr2_updt_u );
  printf( "%% tt_svd:	%le \n", tt_svd_fact + tt_svd_updt_a +
									 tt_svd_updt_uv );
  printf( "%%	tt_svd_fact:	%le \n", tt_svd_fact );
  printf( "%%	tt_svd_updt_a:	%le \n", tt_svd_updt_a );
  printf( "%%	tt_svd_updt_uv:	%le \n", tt_svd_updt_uv );
  printf( "%% total_time:	%le \n", 
		   tt_spl + 
		   tt_qr1_fact + tt_qr1_updt_a + tt_qr1_updt_v + 
		   tt_qr2_fact + tt_qr2_updt_a + tt_qr2_updt_u +
		   tt_svd_fact + tt_svd_updt_a + tt_svd_updt_uv );
  
  printf("%% test_time: %le \n", test_time);
#endif


  return 0;

}

// ========================================================================
// Auxiliary functions 

static void Allocate_work_array( cusolverDnHandle_t handle,
				double ** work_pg_p,
				int m_A, int n_A, double * A_pg, int ldim_A,
				int bl_size ) {
  
  // this function allocates memory for the largest work array needed by 
  // any cusolver function later in rand_utv_gpu
  
  int lwork_qr = 0, lwork_svd = 0;

  // determine largest size needed by a dgeqrf (note dormqr uses the same 
  // size, so we don't need to calculate it)
  cusolverDnDgeqrf_bufferSize( handle, 
				m_A, bl_size, A_pg, ldim_A, 
				& lwork_qr );

  // determine largest size needed by a dgesvd
  cusolverDnDgesvd_bufferSize( handle, bl_size, bl_size, & lwork_svd );
  
  // allocate memory
  cudaError_t cudaStat;
  cudaStat = cudaMalloc( work_pg_p, sizeof( double ) * max( lwork_qr, lwork_svd ) );
  assert( cudaStat == cudaSuccess );
}

// ========================================================================
static void Set_to_identity( int m_A, int n_A, double * A_pc, int ldim_A ) {

  // This function sets contents of matrix A, stored in cpu,
  // to the identity matrix

  int i,j;
  int mn_A = min( m_A, n_A );

  // Set the full matrix to 0
  for ( j=0; j<n_A; j++ ) {
    for ( i=0; i<m_A; i++ ) {
	  A_pc[ i + j * ldim_A ] = 0.0; 
	}
  }

  // Set the main diagonal to 1
  for ( i=0; i < mn_A; i++ ) {
    A_pc[ i + i * ldim_A ] = 1.0;
  }

}

// =========================================================================
static void Normal_random_matrix( int m_A, int n_A,
				double * A_pg, int ldim_A,
				curandGenerator_t rand_gen ) {
  // fills the gpu array with random numbers
  // of standard normal distribution
  curandGenerateNormalDouble( rand_gen, A_pg, m_A * n_A,
								0.0, 1.0 );
}

// ========================================================================
static void local_magqr_nopiv( int m_A, int n_A, 
				double * A_pg, int ldim_A,
				double * tau_h, magma_int_t * magInfo ){
  // given an m_A x n_A matrix A, calculates the QR factorization of A;
  // the HH vectors overwrite the lower tri portion of A
  // T_d holds the info necessary to apply Q to a matrix
  
  magma_dgeqrf2_gpu( m_A, n_A, A_pg, ldim_A, tau_h, magInfo );

}

// ========================================================================
static void local_magormqr( magma_side_t mag_side, magma_trans_t mag_trans,
				int m_C, int n_C, double * C_pg, int ldim_C,
				int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_h,
				double * work_h, double * T_d,
				magma_int_t * magInfo ) {
  // calculates op(Q) * C or C * op(Q), where Q is the HH matrix from
  // the QR factorization of A; the HH vectors used to form Q must
  // be stored in the lower tri part of A_pg
  // 'side' tells you which side of the mult Q is on
  
  // declare aux vars
  int nb, lwork;

  // quick return if possible
  if ( m_C == 0 || n_C == 0 ) {
    return;
  }

  // calculate lwork
  nb = magma_get_dgeqrf_nb( m_A, n_A );
  
  if ( mag_side == MagmaLeft ) {
	lwork = ( m_C - n_A + nb ) * ( n_C * nb ) + n_C * nb;	   
  }
  else { // then mag_side == MagmaRight
    lwork = ( n_C - n_A + nb ) * ( m_C + nb ) +	m_C * nb;
  }

  // do the calculation
  magma_dormqr_gpu( mag_side, mag_trans,
			m_C, n_C, n_A, 
			A_pg, ldim_A,
			tau_h,
			C_pg, ldim_C,
			work_h, lwork, T_d, nb, magInfo );

}
// ========================================================================
static void gpu_orth( int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_h, double * T_d,
				magma_int_t * magInfo ) {
  // given an m_A x n_A matrix A, calculates a matrix Q with ON columns
  // such that A and Q share the same column space; Q overwrites A

  int nb;

  // get block size to use as input later
  nb = magma_get_dgeqrf_nb( m_A, n_A );

  if ( m_A >= n_A ) {
	// compute QR fact, get HH reflectors
	magma_dgeqrf_gpu( m_A, n_A, A_pg, ldim_A,
				  tau_h, T_d, magInfo );

	// form Q
	magma_dorgqr_gpu( m_A, n_A, n_A, A_pg, ldim_A,
				  tau_h, T_d, 
				  nb, magInfo );
  }
  else {
	// compute QR fact, get HH reflectors
	magma_dgeqrf_gpu( m_A, m_A, A_pg, ldim_A,
				  tau_h, T_d, magInfo );

	// form Q
	magma_dorgqr_gpu( m_A, m_A, m_A, A_pg, ldim_A,
				  tau_h, T_d, 
				  nb, magInfo );
  }
}

// ========================================================================
static void gpu_dgeqrf( cusolverDnHandle_t handle,
				int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_pg,
				double * work_pg, int * devInfo ) {
  // given an m_A x n_A matrix A, calculates the QR factorization of A;
  // the HH vectors overwrite the lower tri portion of A
  
  // declare and initialize auxiliary variables
  int lwork = 0; // size of work buffer

  // determine size needed for workspace; this function just sets lwork 
  // to whatever it needs to be

  cusolverDnDgeqrf_bufferSize( handle, 
				m_A, n_A, A_pg, ldim_A,
				& lwork );
  
  // compute factorization
  cusolverDnDgeqrf( handle,
				m_A, n_A, A_pg, ldim_A,
				tau_pg, work_pg, lwork,
				devInfo );

}

__global__
static void Make_upper_tri_kern( int m_A, int n_A,
				int ldim_A, double * A_pg ) {

  int ij = blockIdx.x * blockDim.x + threadIdx.x;
  
  // determine column, row indices for current thread
  int i = ij - m_A * ( ij / m_A ); // floor implicitly taken in div
  int j = ij / m_A; // floor implicitly taken

  // zero out elements below the main diagonal
  if ( ( ij < m_A * n_A ) && ( i > j ) ) {
    A_pg[ i + j * ldim_A ] = 0.0;
  }

}

// ========================================================================
static void Make_upper_tri( int m_A, int n_A,
				int ldim_A, double * A_pg ) {
  // given a matrix stored in device memory, 
  // this function sets all entries below the main diagonal to zero
  Make_upper_tri_kern<<<( m_A*n_A / gpu_thread_size ) + 1, gpu_thread_size >>>( m_A, n_A, ldim_A, A_pg );
}

// ========================================================================
__global__
static void Set_ss_diag_mat_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * ss_pg ) {
  // kernel function which sets matrix represented by A_pg to a diagonal
  // matrix with the svs of A on the diagonal

  int ij = blockIdx.x * blockDim.x + threadIdx.x;
  
  // determine column, row indices for current thread
  int i = ij - m_A * ( ij / m_A ); // floor implicitly taken in div
  int j = ij / m_A; // floor implicitly taken

  // fill in matrix
  if ( ( ij < m_A * n_A ) && ( i == j ) ) {
    A_pg[ i + j * ldim_A ] = ss_pg[ i ];
  }
  else if ( ij < m_A * n_A ) {
    A_pg[ i + j * ldim_A ] = 0.0; 
  }
  
}

// ========================================================================
static void Set_ss_diag_mat( int m_A, int n_A, double * A_pg, int ldim_A,
				double * ss_pg ) {
  // host function which sets matrix represented by A_pg to a diagonal
  // matrix with the svs of A on the diagonal
  Set_ss_diag_mat_kern<<< m_A * n_A / gpu_thread_size + 1, gpu_thread_size >>>( m_A, n_A, A_pg, ldim_A,
				ss_pg );

}

// ========================================================================
static void local_magsvd( int m_A, int n_A, double * A_pg, int ldim_A,
                double * ss_pg,
                double * U_pg, int ldim_U,
                double * Vt_pg, int ldim_Vt,
				double * work_h, int * iwork_h ) {
  // given an m_A x n_A matrix A stored in device memory in A_pg,
  // this function computes the svd on the device

  // declare and initialize auxiliary variables

  double * ss_p;
  double * A_p, * U_p, * Vt_p;

  int lwork = 0; // size of work buffer
  magma_int_t * magInfo = NULL; // stored on host 

  // vars for determining size of work array
  int nb = magma_get_dgesvd_nb( m_A, n_A );
  int A_mx, A_mn;

  // get max and min
  A_mx = max( m_A, n_A );
  A_mn = min( m_A, n_A );

  // allocate space for devInfo on device
  magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );

  // determine size of work array
  // all SVDs for randUTV will be square, which determines our
  // equation for finding lwork
  lwork = 3*A_mn + max( 3*A_mn*A_mn + 4*A_mn, (A_mx+A_mn)*nb );

  // arrays must be in host memory for magma svd
  A_p = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  U_p = ( double * ) malloc( m_A * m_A * sizeof( double ) );
  Vt_p = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  ss_p = ( double * ) malloc( A_mn * sizeof( double ) );

  magma_dgetmatrix( m_A, n_A, A_pg, ldim_A, A_p, m_A );
  magma_dgetmatrix( m_A, m_A, U_pg, ldim_U, U_p, m_A );
  magma_dgetmatrix( n_A, n_A, Vt_pg, ldim_Vt, Vt_p, n_A );

  // compute factorization
  magma_dgesdd( MagmaAllVec,
                m_A, n_A, A_p, m_A,
                ss_p,
                U_p, m_A,
                Vt_p, n_A,
                work_h, lwork, iwork_h, magInfo );

  // transfer results back to device
  magma_dsetmatrix( m_A, m_A, U_p, m_A, U_pg, ldim_U );
  magma_dsetmatrix( n_A, n_A, Vt_p, n_A, Vt_pg, ldim_Vt );
  cudaMemcpy( ss_pg, ss_p, A_mn * sizeof( double ),
				cudaMemcpyHostToDevice );

  // set contents of A_pg to zeros with svs on diagonal
  Set_ss_diag_mat( m_A, n_A, A_pg, ldim_A, ss_pg );

  // free memory
  free( A_p );
  free( U_p );
  free( Vt_p );
  free( ss_p );

  free( magInfo );

}

// ========================================================================
static void local_left_svecs( int m_A, int n_A, double * A_pg, int ldim_A,
				double * work_h, int * iwork_h ) {
  // given an m_A x n_A matrix A stored in device memory in A_pg,
  // this function returns the first min(m_A,n_A) left singular
  // vectors in the first min(m_A,n_A) columns of A

  // declare and initialize auxiliary variables

  double * ss_p;
  double * A_p, * U_p, * Vt_p;

  int lwork = 0; // size of work buffer
  magma_int_t * magInfo = NULL; // stored on host 

  // vars for determining size of work array
  int nb = magma_get_dgesvd_nb( m_A, n_A );
  int A_mx, A_mn;

  // get max and min
  A_mx = max( m_A, n_A );
  A_mn = min( m_A, n_A );

  // allocate space for devInfo on device
  magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );

  // arrays must be in host memory for magma svd
  A_p = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  U_p = ( double * ) malloc( m_A * m_A * sizeof( double ) );
  Vt_p = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  ss_p = ( double * ) malloc( A_mn * sizeof( double ) );

  magma_dgetmatrix( m_A, n_A, A_pg, ldim_A, A_p, m_A );

  // determine size of work array
  magma_dgesdd( MagmaSomeVec,
				m_A, n_A, A_p, ldim_A,
				ss_p,
				U_p, m_A,
				Vt_p, n_A,
				work_h, -1, iwork_h, magInfo );

  lwork = work_h[ 0 ];

  // compute factorization
  magma_dgesdd( MagmaSomeVec,
                m_A, n_A, A_p, m_A,
                ss_p,
                U_p, m_A,
                Vt_p, n_A,
                work_h, lwork, iwork_h, magInfo );

  // transfer results back to device
  magma_dsetmatrix( m_A, A_mn, U_p, m_A, A_pg, ldim_A );

  // free memory
  free( A_p );
  free( U_p );
  free( Vt_p );
  free( ss_p );

  free( magInfo );

}

// ========================================================================
static void gpu_dgesvd( cusolverDnHandle_t handle, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				double * ss_pg,
				double * U_pg, int ldim_U,
				double * Vt_pg, int ldim_Vt,
				double * work_pg, int * devInfo ) {
  // given an m_A x n_A matrix A stored in device memory in A_pg,
  // this function computes the svd on the device
  
  // declare and initialize auxiliary variables
  int lwork = 0; // size of work buffer
  double * rwork_pg = NULL; // if dgesvd fails to converge, contains
						 // unconverged superdiagonal elements
  signed char a = 'A';

  // determine size of work array
  cusolverDnDgesvd_bufferSize( handle,
				m_A, n_A, 
				& lwork );

  // compute factorization
  cusolverDnDgesvd( handle, 
				a, a,
				m_A, n_A, A_pg, ldim_A,
				ss_pg, 
				U_pg, ldim_U,
				Vt_pg, ldim_Vt,
				work_pg, lwork, rwork_pg, devInfo );

  // set contents of A_pg to zeros with svs on diagonal
  Set_ss_diag_mat( m_A, n_A, A_pg, ldim_A, ss_pg );

}

// ========================================================================
static void gpu_dgemm( cublasHandle_t handle, 
				char opA, char opB, double alpha, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				int m_B, int n_B, double * B_pg, int ldim_B,
				double beta,
				int m_C, int n_C, double * C_pg, int ldim_C ) {

  // generate the correct transpose option identifier that CUBLAS accepts
  // also determine the correct "middle" dim of the mult
  cublasOperation_t cutransA, cutransB;
  cublasStatus_t cublasStat = CUBLAS_STATUS_SUCCESS;
  int middle_dim;

  if ( opA == 'N' ) { cutransA = CUBLAS_OP_N; middle_dim = n_A; }
  else if ( opA == 'T' ) { cutransA = CUBLAS_OP_T; middle_dim = m_A; }

  if ( opB == 'N' ) { cutransB = CUBLAS_OP_N; }
  else if ( opB == 'T' ) { cutransB = CUBLAS_OP_T; }


  // do the multiplication
  cublasStat = cublasDgemm( handle, cutransA, cutransB,
				m_C, n_C, middle_dim, & alpha,
				A_pg, ldim_A, B_pg, ldim_B, 
				& beta, C_pg, ldim_C );
	
  assert( cublasStat == CUBLAS_STATUS_SUCCESS );
}

// =======================================================================
static void magma_dlarft( cublasHandle_t handle, int n, int k, 
				double * V_pg, int ldim_V,
				double * tau_h,
				double * T_pg, int ldim_T,
				double * TVt_pg, int ldim_TVt ) {
  // forms the middle matrix T in a UT representation of a HH matrix
  //
  // n: order of HH matrix (in our case, its number of rows)
  // k: number of HH reflectors (in our case, its number of cols)
  // V_pg: array containing the HH reflectors in its lower triangular
  //       portion (V will usually be the output of dgeqrf)
  // tau_pg: tau_pg[i] contains the scalar factor of the HH reflector H(i)
  // T_pg: array, dimension k x k; this matrix is the output

  // declare aux vars
  double * V_h, * T_h;
  double d_one = 1.0, d_zero = 0.0;

  // create temporary host storage arrays; dlarft will be executed on
  // the host
  T_h = ( double * ) malloc( k * k * sizeof( double ) );
  V_h = ( double * ) malloc( n * k * sizeof( double ) );

  // transfer info to host storage
  magma_dgetmatrix( n, k, V_pg, ldim_V, V_h, n );

  // perform dlarft
  lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
				& n, & k, 
				V_h, & n, 
				tau_h, 
				T_h, & k ); 

  // transfer info to device storage
  magma_dsetmatrix( k, k, T_h, k, T_pg, ldim_T );

  // form matrix TV'
  Make_upper_tri( k, k, ldim_T, T_pg );
  cublasDtrmm( handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, 
				CUBLAS_OP_T, CUBLAS_DIAG_UNIT, 
				k, k, 
				& d_one,
				V_pg, ldim_V,
				T_pg, ldim_T,
				TVt_pg, ldim_TVt );
  
  cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T,
				k, n-k, k,
				& d_one,
				T_pg, ldim_T,
				& V_pg[ k + 0 * ldim_V ], ldim_V,
				& d_zero,
				& TVt_pg[ 0 + k * ldim_TVt ], ldim_TVt );

  // free memory
  free( T_h );
  free( V_h );
}

// =======================================================================
static void dlarft_gpu( cublasHandle_t handle, int n, int k, 
				double * V_pg, int ldim_V,
				double * tau_pg,
				double * T_pg, int ldim_T,
				double * TVt_pg, int ldim_TVt,
				double * work_pg ) {
  // forms the middle matrix T in a UT representation of a HH matrix
  //
  // n: order of HH matrix (i.e. its number of rows)
  // k: number of HH reflectors
  // V_pg: array containing the HH reflectors in its lower triangular
  //       portion (V will usually be the output of dgeqrf)
  // tau_pg: tau_pg[i] contains the scalar factor of the HH reflector H(i)
  // T_pg: array, dimension k x k; this matrix is the output
  
  // declaration of auxiliary vars
  int i;
  double d_one = 1.0, d_zero = 0.0;
  double alpha;
  cublasStatus_t status;

  // replace diagonal of V with ones; store original entries to restore
  // later
  replace_diag_ones( n, k, V_pg, ldim_V,
				work_pg );

  // set diagonal entries of T to entries of tau
  replace_diag_vec( k, k, T_pg, ldim_T, tau_pg );
  
  // build T
  for ( i=0; i<k; i++ ) {
    
	// check for 0.0 values in tau (==> H(i) = I) and fill in what
	// values of T we can accordingly
	check_tau_zeroes( k, k, T_pg, ldim_T,
					tau_pg );
    
	// T(1:i-1,i) = -tau(i) * V(i:n,1:i-1)'*V(i:n,i)
    cudaMemcpy( & alpha, & tau_pg[ i ], sizeof( double ), cudaMemcpyDeviceToHost );
	alpha = - alpha;
	cublasDgemv( handle, CUBLAS_OP_T,
				n - i, i, 
				& alpha,
				& V_pg[ i + 0 * ldim_V ], ldim_V,
				& V_pg[ i + i * ldim_V ], 1,
				& d_zero,
				& T_pg[ 0 + i * ldim_T ], 1 );
	
	// T(1:i-1,i) = T(1:i-1,1:i-1) * T(1:i-1,i)
	cublasDtrmv( handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
				CUBLAS_DIAG_NON_UNIT,
				i, T_pg, ldim_T,
				& T_pg[ 0 + i * ldim_T ], 1 );
  }
  
  // restore diagonal entries of V
  replace_diag_vec( n, k, V_pg, ldim_V,
				work_pg );

  // form T * V'
  Make_upper_tri( k, k, ldim_T, T_pg );
  cublasDtrmm( handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, 
				CUBLAS_OP_T, CUBLAS_DIAG_UNIT, 
				k, k, 
				& d_one,
				V_pg, ldim_V,
				T_pg, ldim_T,
				TVt_pg, ldim_TVt );
  
  cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T,
				k, n-k, k,
				& d_one,
				T_pg, ldim_T,
				& V_pg[ k + 0 * ldim_V ], ldim_V,
				& d_zero,
				& TVt_pg[ 0 + k * ldim_TVt ], ldim_TVt );
}

// =======================================================================
__global__
static void replace_diag_ones_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * work_pg ) {
  // declaration of aux vars
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int mn_A;

  // TODO: not sure if macro works in device function? So doing min the long way
  if ( m_A > n_A ) {
    mn_A = n_A;
  }
  else {
    mn_A = m_A;
  }

  // replace the diagonal elements of A with 1.0, store elements
  // in work_pg
  if ( i < mn_A ) {
	work_pg[ i ] = A_pg[ i + i * ldim_A ];
	A_pg[ i + i * ldim_A ] = 1.0;
  }

}

static void replace_diag_ones( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * work_pg ) {
  // this function replaces the diagonal elements of A with ones and
  // stores the entries in work_pg for later retrieval

  int mn_A = min( m_A, n_A );

  replace_diag_ones_kern
				<<< mn_A / gpu_thread_size + 1, gpu_thread_size >>> 
				( m_A, n_A, A_pg, ldim_A,
				work_pg );
  
}

// =======================================================================
__global__
static void replace_diag_vec_kern( int m_A, int n_A, 
				double * A_pg, int ldim_A,
				double * vec_pg ) {
  // declaration of aux vars
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int mn_A;

  // TODO: not sure if macro works in device function? So doing min the long way
  if ( m_A > n_A ) {
    mn_A = n_A;
  }
  else {
    mn_A = m_A;
  }

  // replace the diagonal elements of A with the elements of vec_pg 
  if ( i < mn_A ) {
	A_pg[ i + i * ldim_A ] = vec_pg[ i ];
  }


}

static void replace_diag_vec( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * vec_pg ) {
  // this function replaces the diagonal elements of A with the elements
  // stored in vec_pg

  int mn_A = min( m_A, n_A );

  replace_diag_vec_kern
				<<< mn_A / gpu_thread_size + 1, gpu_thread_size >>>
				( m_A, n_A, A_pg, ldim_A,
				vec_pg );
}

// =======================================================================
__global__
static void check_tau_zeroes_kern( int m_T, int n_T,
				double * T_pg, int ldim_T,
				double * tau_pg ) {
  // declaration/initialization of aux vars
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;

  if ( i < m_T && tau_pg[ i ] == 0.0 ) {
    for ( j=0; j<i; j++ ) {
	  T_pg[ j + i * ldim_T ] = 0.0;
	}
  }

}

static void check_tau_zeroes( int m_T, int n_T,
				double * T_pg, int ldim_T,
				double * tau_pg ) {
  // this function checks for zeroes in tau_pg (==> H(i) = I)
  // and, if it finds any, sets the necessary entries of T to 0.0
  check_tau_zeroes_kern
				<<< m_T / gpu_thread_size + 1, gpu_thread_size >>>
				( m_T, n_T, T_pg, ldim_T,
				tau_pg );
}

// =======================================================================
static void my_dormqr_gpu( cublasHandle_t handle,
				char side, char trans, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				int m_B, int n_B, double * B_pg, int ldim_B,
				double * TVt_pg, int ldim_TVt,
				double * work_pg ) {
  // applies the HH matrix H = I - VTV' to B
  // V is the lower triangular matrix with ones on the diagonal
  // and the HH reflectors stored in the columns
  // A_pg: matrix A after a call to dgeqrf; it holds the HH reflectors
  //	   in its lower triangular part

  // auxiliary vars
  int m_work, n_work, ldim_work;
  cublasOperation_t cublas_op;
  double * R_pg;
  int ldim_R = n_A;
  int k = n_A; // this is the number of HH reflectors 
  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0;

  double * ones_vc_d, * zeros_vc_d;

  cudaMalloc( ( void ** ) & R_pg, n_A * n_A * sizeof( double ) );
  cudaMalloc( ( void ** ) & ones_vc_d, n_A * sizeof( double ) );
  cudaMalloc( ( void ** ) & zeros_vc_d, n_A * sizeof( double ) );

  // initialize vectors of ones and zeros
  set_vc_one( n_A, ones_vc_d );
  set_vc_zero( n_A, zeros_vc_d );

  // quick exit if possible
  if ( m_B == 0 || n_B == 0 ) {
    return;
  }

  // change A_pg into V needed for mult; store upper triangular part in
  // R so we can restore it later
  magma_dcopymatrix( n_A, n_A, A_pg, ldim_A, R_pg, ldim_R );

  Make_upper_tri( n_A, n_A, ldim_R, R_pg );

  mm_A_minus_B( n_A, n_A, A_pg, ldim_A, R_pg, ldim_R );
 
  replace_diag_vec( m_A, n_A, A_pg, ldim_A, ones_vc_d );

  // begin multiplication

  if ( side == 'L' || side == 'l' ) { // compute H * B or H' * B
	    
	// in this function, we only support computing H' * B if side == L,
	// i.e. trans == T

	// compute H' * B
	
    ldim_work = k;

	// work <-- V' * B
	//cublasDtrmm( handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
	//				CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
	//				m_B, n_B,
	//				& d_one,
	//				A_pg, ldim_A,
	//				B_pg, ldim_B,
	//				work_pg, ldim_work );
	cublasDgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N,
					k, n_B, m_B,
					& d_one,
					A_pg, ldim_A,
					B_pg, ldim_B,
					& d_zero,
					work_pg, ldim_work );	

	// B <-- - TVt' * work + B;
    cublasDgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, 
					m_B, n_B, k, 
					& d_neg_one, 
					TVt_pg, ldim_TVt, 
					work_pg, ldim_work, 
					& d_one, 
					B_pg, ldim_B );
	
  }
  else { // then side == 'R'; compute B * H or B * H'
	
	ldim_work = m_B;

	// this function only supports computing B * H if side == R,
	// i.e. trans == N

	// compute B * H

    // work <-- B * V
    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
					m_B, k, n_B,
					& d_one,
					B_pg, ldim_B,
					A_pg, ldim_A,
					& d_zero,
					work_pg, ldim_work );

    // B <-- - work * TVt + B
    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, 
					m_B, n_B, k, 
					& d_neg_one, 
					work_pg, ldim_work,
					TVt_pg, ldim_TVt,
					& d_one,
					B_pg, ldim_B );
  
  }

  // restore the upper diagonal part of A
  replace_diag_vec( m_A, n_A, A_pg, ldim_A, zeros_vc_d );

  mm_A_plus_B( n_A, n_A, A_pg, ldim_A, R_pg, ldim_R );

  // free memory
  cudaFree( R_pg );
  cudaFree( ones_vc_d );
  cudaFree( zeros_vc_d );

  // TODO: add checks to always make sure work_pg is large enough; will have to pass in size
  // of work_pg

}

// ========================================================================
__global__
static void mm_A_minus_B_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * B_pg, int ldim_B ) {
  // for matrices A and B, computes A <-- A - B
 
  // aux vars
  int ij = blockIdx.x * blockDim.x + threadIdx.x;
  
  // determine column, row indices for current thread
  int i = ij - m_A * ( ij / m_A ); // floor implicitly taken in div
  int j = ij / m_A; // floor implicitly taken
 
  // do the subtraction
  if ( i < m_A && j < n_A ) {
    A_pg[ i + j * ldim_A ] = A_pg[ i + j * ldim_A ] - B_pg[ i + j * ldim_B ];
  }

}

static void mm_A_minus_B( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * B_pg, int ldim_B ) {

  mm_A_minus_B_kern<<< m_A * n_A / gpu_thread_size + 1, gpu_thread_size >>>
				( m_A, n_A, A_pg, ldim_A,
				B_pg, ldim_B );

}

// ========================================================================
__global__
static void mm_A_plus_B_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * B_pg, int ldim_B ) {
  // for matrices A and B, computes A <-- A + B
 
  // aux vars
  int ij = blockIdx.x * blockDim.x + threadIdx.x;
  
  // determine column, row indices for current thread
  int i = ij - m_A * ( ij / m_A ); // floor implicitly taken in div
  int j = ij / m_A; // floor implicitly taken
 
  // do the subtraction
  if ( i < m_A && j < n_A ) {
    A_pg[ i + j * ldim_A ] = A_pg[ i + j * ldim_A ] + B_pg[ i + j * ldim_B ];
  }

}

static void mm_A_plus_B( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * B_pg, int ldim_B ) {

  mm_A_plus_B_kern<<< m_A * n_A / gpu_thread_size + 1, gpu_thread_size >>>
				( m_A, n_A, A_pg, ldim_A,
				B_pg, ldim_B );

}

// ========================================================================
__global__
static void set_vc_zero_kern( int lvec, double * vec_pg ) {
  // sets entries in vec_pg (stored on device) to zero

  // aux vars
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // do the subtraction
  if ( i < lvec ) {
    vec_pg[ i ] = 0.0;
  }

}

static void set_vc_zero( int lvec, double * vec_pg ) {

  set_vc_zero_kern<<< lvec / gpu_thread_size + 1, gpu_thread_size >>>
				( lvec, vec_pg );

}

// ========================================================================
__global__
static void set_vc_one_kern( int lvec, double * vec_pg ) {
  // sets entries in vec_pg (stored on device) to one

  // aux vars
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // do the subtraction
  if ( i < lvec ) {
    vec_pg[ i ] = 1.0;
  }

}

static void set_vc_one( int lvec, double * vec_pg ) {

  set_vc_one_kern<<< lvec / gpu_thread_size + 1, gpu_thread_size >>>
				( lvec, vec_pg );

}
