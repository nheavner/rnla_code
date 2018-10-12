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

#define PROFILE
#define PROFILE_FOR_GRAPHING

// =======================================================================
// Definition of global variables

static int gpu_thread_size = 256; // TODO: temporary value; determine how to optimize
						 // this gives the number of threads per block for
						 // kernel calls

// =======================================================================
// TODO: delete this function once we're finished building
static void gpu_print_double_matrix( char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A ) {
  int  i, j;

  double * A_pc;

  A_pc = ( double * ) malloc( m_A * n_A * sizeof( double ) );

  magma_dgetmatrix( m_A, n_A, buff_A, ldim_A, A_pc, m_A );

  cudaDeviceSynchronize();
  printf( "%s = [\n", name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%.16e ", A_pc[ i + j * m_A ] );
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

}
// ========================================================================
// Declaration of local prototypes

static void calculate_work_array_size( int bl_size, int pp,
				int * lwork, int * iwork_h, magma_int_t * magInfo );

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

static void local_left_svecs_v2( cublasHandle_t cublasH,
				int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_h,
				double * work_h, int * iwork_h,
				double * T_d, int ldim_T,
				double * TVt_d, int ldim_TVt ); 

static void project_away( cublasHandle_t cublasH,
				int m_A, int n_A, double * A_d, int ldim_A,
				int m_V, int n_V, double * V_d, int ldim_V ); 

static void gpu_dgemm( cublasHandle_t handle,
				char opA, char opB, double alpha, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				int m_B, int n_B, double * B_pg, int ldim_B,
				double beta,
				int m_C, int n_C, double * C_pg, int ldim_C );

static void magma_dlarft( cublasHandle_t handle, int n, int k, 
				double * V_pg, int ldim_V,
				double * tau_h,
				double * T_pg, int ldim_T,
				double * TVt_pg, int ldim_TVt );

__global__
static void replace_diag_vec_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * vec_pg );

static void replace_diag_vec( int m_A, int n_A,
				double * A_pg, int ldim_A,
				double * vec_pg );

static void my_dormqr_gpu( cublasHandle_t handle,
				char side, char trans, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				int m_B, int n_B, double * B_pg, int ldim_B,
				double * TVt_pg, int ldim_TVt ); 

static void oversample_downdate_gpu( cublasHandle_t handle,
				int m_Vloc, int n_Vloc, double * Vloc_pg, int ldim_Vloc,
				int m_Y_os, int n_Y_os, double * Y_os_pg, int ldim_Y_os,
				double * TVt_pg, int ldim_TVt );
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

static timespec start_timer( void );

static double stop_timer( timespec t1 );
// ========================================================================

// Main function
int rand_utv_gpu( 
		int m_A, int n_A, double * A_pc, int ldim_A,
		int build_U, int m_U, int n_U, double * U_pc, int ldim_U,
		int build_V, int m_V, int n_V, double * V_pc, int ldim_V,
		int bl_size, int pp, int q_iter ) {

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
  int i, j;
  int mn_A, num_spl;
  int m_G, n_G, ldim_G, m_G_loc, n_G_loc,
	  m_Y, n_Y, ldim_Y, m_Y_loc, n_Y_loc;
  int m_Y_svecs, n_Y_svecs, ldim_Y_svecs;
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
  int ldim_T; // holds the T matrix in a UT representation of
						// HH matrices
  int ldim_TVt;
  double * A_pg, * U_pg, * V_pg; // pointers to matrix arrays in gpu
  double * G_pg, * G_loc_pg, * A_loc_pg, * A_right_pg, * A_bl_pg, * A_BR_pg,
		 * A_22_pg, * A_12_pg, * A_23_pg, * Y_pg, * Y_loc_pg, * Y_svecs_pg;
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

  int * devInfo = NULL; // another var for cusolver functions

  // work arrays for magma functions
  double * work_h;
  int * iwork_h;
  int lwork_mx; // for determining size of work array
  magma_int_t * magInfo;
  double * T_d; // T matrix for applying HH matrices
  double * tau_h; // tau vector for applying HH matrices

#ifdef PROFILE
  double tt_spl,
		 tt_qr1_fact, tt_qr1_updt_a, tt_qr1_updt_v,
		 tt_qr2_fact, tt_qr2_updt_a, tt_qr2_updt_u,
		 tt_svd_fact, tt_svd_updt_a, tt_svd_updt_uv;
  
  timespec time1; // var for timing
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
  tt_spl		 = 0.0;
  tt_qr1_fact    = 0.0;
  tt_qr1_updt_a  = 0.0;
  tt_qr1_updt_v  = 0.0;
  tt_qr2_fact    = 0.0;
  tt_qr2_updt_a  = 0.0;
  tt_qr2_updt_u  = 0.0;
  tt_svd_fact    = 0.0;
  tt_svd_updt_a  = 0.0;
  tt_svd_updt_uv = 0.0;
#endif

  // initialize auxiliary variables
  mn_A = min( m_A, n_A );

  m_A_loc = m_A;	n_A_loc = n_A;
  m_G     = m_A;	n_G     = bl_size + pp;		ldim_G = m_A;
  m_Y     = n_A;	n_Y     = bl_size + pp;		ldim_Y = n_A;

  m_Y_svecs = bl_size + pp;		n_Y_svecs = bl_size + pp;	ldim_Y_svecs = m_Y_svecs;

  m_Vt_svd = bl_size;	n_Vt_svd = bl_size;		ldim_Vt_svd = bl_size;
  m_U_svd  = bl_size;	n_U_svd  = bl_size;		ldim_U_svd  = bl_size;

  ldim_T   = bl_size + pp;
  ldim_TVt = bl_size + pp;

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

  cudaStat = cudaMalloc( & Y_svecs_pg, ( bl_size + pp ) * ( bl_size + pp ) * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  cudaStat = cudaMalloc( & Vt_svd_pg, bl_size * bl_size * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & U_svd_pg, bl_size * bl_size * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  cudaStat = cudaMalloc( & tau_pg, bl_size * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & T_pg, ( bl_size + pp ) * ( bl_size + pp ) * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & TVt_pg, m_A * ( bl_size + pp ) * sizeof( double ) );
  // TODO: should previous line replace m_A with bl_size + pp?
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & ss_pg, ( bl_size + pp ) * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  cudaStat = cudaMalloc( & Tmp_pg, ( m_A * ( bl_size + pp ) ) * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  cudaMalloc( ( void ** ) & devInfo, sizeof( int ) );
  
  // determine max size of work array for magma svd calcs,
  // allocate memory for the arrays
  mn_A = min( m_A, n_A );

  magma_imalloc_pinned( & iwork_h, 8 * ( bl_size + pp ) );
  magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );

  calculate_work_array_size( bl_size, pp, & lwork_mx, iwork_h, magInfo );
  magma_dmalloc_pinned( & work_h, lwork_mx );


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
	
	m_A_loc   = m_A - i;	n_A_loc   = n_A - i;
	m_G_loc   = m_A - i;	n_G_loc   = num_spl + pp;
    m_Y_loc   = n_A - i;	n_Y_loc   = min( num_spl + pp, m_Y_loc );
    m_A_right = m_A;	    n_A_right = n_A - i;
	m_V_right = n_A;	    n_V_right = n_A - i;
	m_A_bl    = m_A - i;	n_A_bl    = num_spl;
	m_A_BR    = m_A - i;	n_A_BR    = n_A - i - num_spl;
	m_U_right = m_U;	    n_U_right = n_U - i;
	m_A_22    = num_spl;	n_A_22    = num_spl;
	m_Vt_svd  = num_spl;	n_Vt_svd  = num_spl;
	m_U_svd   = num_spl;	n_U_svd   = num_spl;
	m_A_12    = i;			n_A_12    = num_spl;
	m_A_23    = num_spl;	n_A_23    = n_A - i - num_spl;
	m_U_mid   = m_U;		n_U_mid   = num_spl;
	m_V_mid   = m_V;		n_V_mid   = num_spl;
    
	A_loc_pg   = & A_pg[ i + i * ldim_A ];
	A_right_pg = & A_pg[ 0 + i * ldim_A ];
	G_loc_pg   = & G_pg[ i + 0 * ldim_G ];
    Y_loc_pg   = & Y_pg[ i + 0 * ldim_Y ]; 
    V_right_pg = & V_pg[ 0 + i * ldim_V ];
	A_bl_pg    = & A_pg[ i + i * ldim_A ];
	A_BR_pg    = & A_pg[ i + ( i + num_spl ) * ldim_A ];
	U_right_pg = & U_pg[ 0 + i * ldim_U ];
	A_22_pg    = & A_pg[ i + i * ldim_A ];
    A_12_pg    = & A_pg[ 0 + i * ldim_A ];
	A_23_pg    = & A_pg[ i + ( i + num_spl ) * ldim_A ];
	U_mid_pg   = & U_pg[ 0 + i * ldim_U ];
	V_mid_pg   = & V_pg[ 0 + i * ldim_V ];

	// Compute the "sampling" matrix Y
	// Aloc = T([J2,I3],[J2,I3]);
	// Y = Aloc' * randn(m-(i-1)*b,b+p);
#ifdef PROFILE
	time1 = start_timer();
#endif
      // Create random matrix for sampling
      Normal_random_matrix( m_G, n_G, G_pg, ldim_G,
							rand_gen );
		// TODO: this currently fills the ENTIRE array G each loop;
		// unecessary, but does it matter? decide

      // carry out "sampling" multiplication
	  if ( pp > 0 && i > 0 ) {
		gpu_dgemm( cublasH, 
				   t, n, d_one,
				   m_A_loc, n_A_loc, A_loc_pg, ldim_A,
				   m_G_loc, max(n_G_loc - pp,0), & G_loc_pg[0+pp*ldim_G], ldim_G,
				   d_zero,
				   m_Y_loc, max(n_Y_loc - pp,0), & Y_loc_pg[0+pp*ldim_Y], ldim_Y );
	  }
	  else {
		gpu_dgemm( cublasH,
				   t, n, d_one,
				   m_A_loc, n_A_loc, A_loc_pg, ldim_A,
				   m_G_loc, n_G_loc, G_loc_pg, ldim_G,
			       d_zero,
				   m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y );
	  }

	// perform "power iteration" if requested
	// for i_iter = 1:q_iter:
	//   Y = Aloc' * (Aloc * Y);
	// end
    for( j=0; j<q_iter; j++ ) {
	 
	  // TODO: can simplify this conditional?
      if ( pp > 0 ) {
	   
        if ( j < q_iter - 1 ) {
		  if ( i == 1 ) {
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
		  else {
			// reuse G_loc for storage; G <-- A*Y
			gpu_dgemm( cublasH, 
				       n, n, d_one,
					   m_A_loc, n_A_loc, A_loc_pg, ldim_A,
					   m_Y_loc, max(n_Y_loc - pp,0), & Y_loc_pg[0+pp*ldim_Y], ldim_Y,
					   d_zero, 
					   m_G_loc, max(n_G_loc - pp,0), & G_loc_pg[0+pp*ldim_G], ldim_G );
			
			// complete iteration; Y <-- A'*G
			gpu_dgemm( cublasH, 
					   t, n, d_one,
					   m_A_loc, n_A_loc, A_loc_pg, ldim_A,
					   m_G_loc, max(n_G_loc - pp,0), & G_loc_pg[0+pp*ldim_G], ldim_G,
					   d_zero,
					   m_Y_loc, max(n_Y_loc - pp,0), & Y_loc_pg[0+pp*ldim_Y], ldim_Y );
	      }
		}

		else {
          
		  // orthogonalize cols of Y
		  if ( i > 0 ) {
		    // can save some work by avoiding full QR

		    // Y2 = Y2 - Y1*Y1'*Y2;
			project_away( cublasH,
						  m_Y_loc, max(n_Y_loc - pp,0), & Y_loc_pg[ 0 + pp * ldim_Y ], ldim_Y,
						  m_Y_loc, pp, Y_loc_pg, ldim_Y ); 


			// Y2 = orth(Y2);
			gpu_orth( m_Y_loc, max(n_Y_loc - pp,0), & Y_loc_pg[ 0 + pp * ldim_Y ], ldim_Y,
					  tau_h, T_d,
					  magInfo );
		  }
		  else { // i == 0
		    // do QR on all cols of Y
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
	  }
	  else {
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
	}


#ifdef PROFILE
    tt_spl += stop_timer( time1 );
	time1 = start_timer();
#endif
  
    // Construct the local transform to be applied "from the right".
    // if (p > 0)
    //   [~,~,Jtmp] = qr(Y,0);
    //   [Vloc,~,~] = qr(Y(:,Jtmp(1:b)));
    // else
    //   [Vloc,~]   = LOCAL_nonpiv_QR(Y,b);
    // end

	local_magqr_nopiv( m_Y_loc, n_Y_loc, 
					   Y_loc_pg, ldim_Y,
					   tau_h, magInfo ); 

#ifdef PROFILE
    tt_qr1_fact += stop_timer( time1 );
    time1 = start_timer();
#endif

	// construct "TU'" matrix for UT representation of HH matrix
	magma_dlarft( cublasH, m_Y_loc, n_Y_loc, 
				  Y_loc_pg, ldim_Y,
				  tau_h,
				  T_pg, ldim_T,
				  TVt_pg, ldim_TVt );

    // Apply the pivot matrix to rotate maximal mass into the "J2" column
	// T(:,[J2,J3]) = T(:[J2,J3])*Vloc;

    my_dormqr_gpu( cublasH,
				   r, n, 
				   m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y,
				   m_A_right, n_A_right, A_right_pg, ldim_A,
				   TVt_pg, ldim_TVt );


#ifdef PROFILE
    tt_qr1_updt_a += stop_timer( time1 );
	time1 = start_timer();
#endif

    // Update matrix V with transformations from the first QR.
	my_dormqr_gpu( cublasH,
				   r,n,
				   m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y,
				   m_V_right, n_V_right, V_right_pg, ldim_V,
				   TVt_pg, ldim_TVt );

	// if oversampling is done, finish SVD of sampling matrix
	// (most work is done after QR) and update V and A accordingly
	if ( pp > 0 && ( n_A - i - bl_size ) > pp ) {
	  
	  magma_dcopymatrix( n_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y, Y_svecs_pg, ldim_Y_svecs ); 

      local_left_svecs( m_Y_svecs, n_Y_svecs, Y_svecs_pg, ldim_Y_svecs,
				work_h, iwork_h );

      // update A and V with info from svecs of Y
	  gpu_dgemm( cublasH,
			   n, n, d_one,
		   	   m_A_right, bl_size + pp, A_right_pg, ldim_A,
			   m_Y_svecs, n_Y_svecs, Y_svecs_pg, ldim_Y_svecs,
			   d_zero, 
			   m_A_right, n_Y_svecs, Tmp_pg, m_A_right );

	  magma_dcopymatrix( m_A_right, n_Y_svecs, Tmp_pg, m_A_right, A_right_pg, ldim_A ); 

	  gpu_dgemm( cublasH,
			   n, n, d_one,
		   	   m_V_right, n_Y_svecs, V_right_pg, ldim_V,
			   m_Y_svecs, n_Y_svecs, Y_svecs_pg, ldim_Y_svecs,
			   d_zero,
			   m_V_right, n_Y_svecs, Tmp_pg, m_V_right );
	  
	  magma_dcopymatrix( m_V_right, n_Y_svecs, Tmp_pg, m_V_right, V_right_pg, ldim_V ); 


      // downdate oversamples from Y to use in the next sampling matrix 
	  magma_dcopymatrix( m_Y_svecs, n_Y_svecs - bl_size, 
					& Y_svecs_pg[0 + bl_size*ldim_Y_svecs], ldim_Y_svecs, 
					Tmp_pg, m_Y_svecs ); 

      my_dormqr_gpu( cublasH,
	  			     l, t, 
				     m_Y_loc, n_Y_loc, Y_loc_pg, ldim_Y,
				     m_Y_svecs, n_Y_svecs - bl_size, 
					 & Y_svecs_pg[0 + bl_size * ldim_Y_svecs ], ldim_Y_svecs,
				     TVt_pg, ldim_TVt );
	  
	  gpu_dgemm( cublasH,
			     t, n, d_one,
		   	     m_Y_svecs, n_Y_svecs - bl_size, Tmp_pg, m_Y_svecs,
			     m_Y_svecs, n_Y_svecs - bl_size, 
				 & Y_svecs_pg[ 0 + bl_size*ldim_Y_svecs ], ldim_Y_svecs,
			     d_zero,
			     m_Y_loc - bl_size, pp,
				 & Y_loc_pg[bl_size + 0*ldim_Y], ldim_Y );
	
	}

#ifdef PROFILE
	tt_qr1_updt_v += stop_timer( time1 );
	time1 = start_timer();
#endif
    
	// %%% Next determine the rotations to be applied "from the left".
    // [Uloc,Dloc]      = LOCAL_nonpiv_QR(T([J2,I3],J2));

	local_magqr_nopiv( m_A_bl, n_A_bl, 
					   A_bl_pg, ldim_A,
					   tau_h, magInfo );

#ifdef PROFILE
	tt_qr2_fact += stop_timer( time1 );
	time1 = start_timer();
#endif
	
	magma_dlarft( cublasH, m_A_bl, n_A_bl, 
				  A_bl_pg, ldim_A,
				  tau_h,
				  T_pg, ldim_T,
				  TVt_pg, ldim_TVt );

	// update rest of matrix A with transformations from the second QR
	my_dormqr_gpu( cublasH,
				   l, t,
				   m_A_bl, n_A_bl, A_bl_pg, ldim_A,
				   m_A_BR, n_A_BR, A_BR_pg, ldim_A,
				   TVt_pg, ldim_TVt );

#ifdef PROFILE
	tt_qr2_updt_a += stop_timer( time1 );
	time1 = start_timer();
#endif

	// update matrix U with transformations from the second QR
	my_dormqr_gpu( cublasH,
				   r, n,
				   m_A_bl, n_A_bl, A_bl_pg, ldim_A,
				   m_U_right, n_U_right, U_right_pg, ldim_U,
				   TVt_pg, ldim_TVt );

#ifdef PROFILE
	tt_qr2_updt_u += stop_timer( time1 );
	time1 = start_timer();
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
	
	local_magsvd( m_A_22, n_A_22, A_22_pg, ldim_A,
				  ss_pg, U_svd_pg, ldim_U_svd,
				  Vt_svd_pg, ldim_Vt_svd,
				  work_h, iwork_h );

#ifdef PROFILE
	tt_svd_fact += stop_timer( time1 );
	time1 = start_timer();
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
	tt_svd_updt_a += stop_timer( time1 );
	time1 = start_timer();
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
	tt_svd_updt_uv += stop_timer( time1 );
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
  cudaFree( Y_svecs_pg );
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

  cudaFree( devInfo );

#ifdef PROFILE
  #ifdef PROFILE_FOR_GRAPHING
  printf("n = %d \n", n_A );
  printf("%le %le %le %le %le %le \n", tt_spl, 
							   tt_qr1_fact,//+tt_qr1_updt_a+tt_qr1_updt_v, 
							   tt_qr1_updt_a,
							   tt_qr2_fact,//+tt_qr2_updt_a+tt_qr2_updt_u, 
							   tt_qr2_updt_a,
							   tt_svd_fact);//+tt_svd_updt_a+tt_svd_updt_uv);
  #else
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
  #endif
#endif


  return 0;

}

// ========================================================================
// Auxiliary functions 

static void calculate_work_array_size( int bl_size, int pp,
				int * lwork, int * iwork_h, magma_int_t * magInfo ) {
  // finds the longest length needed for the work array work_h used in
  // MAGMA's svd implementation
  //
  // the largest SVD we'll ever do is for a matrix with
  // ( bl_size+pp ) x ( bl_size+pp ) entries
  
  int max_length1, max_length2, max_length3;
  double * work_tmp;

  work_tmp = ( double * ) malloc( sizeof( double ) );

  // length test 1
  magma_dgesdd( MagmaSomeVec,
				bl_size + pp, bl_size + pp, work_tmp, bl_size + pp,
				work_tmp,
				work_tmp, bl_size + pp,
				work_tmp, bl_size + pp,
				work_tmp, -1, iwork_h , magInfo );

  max_length1 = ( int ) work_tmp[ 0 ];

  // length test 2
  magma_dgesdd( MagmaOverwriteVec,
				bl_size + pp, bl_size + pp, work_tmp, bl_size + pp,
				work_tmp,
				work_tmp, bl_size + pp,
				work_tmp, bl_size + pp,
				work_tmp, -1, iwork_h, magInfo );

  max_length2 = ( int ) work_tmp[ 0 ];

  // length test 3
  magma_dgesdd( MagmaAllVec,
				bl_size + pp, bl_size + pp, work_tmp, bl_size + pp,
				work_tmp,
				work_tmp, bl_size + pp,
				work_tmp, bl_size + pp,
				work_tmp, -1, iwork_h, magInfo );

  max_length3 = ( int ) work_tmp[ 0 ];

  * lwork = max( max_length1, max_length2 );

  * lwork = max( * lwork, max_length3 );

  free( work_tmp );
 
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

// =========================================================================
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
  int A_mn;

  // get max and min
  A_mn = min( m_A, n_A );

  // allocate space for devInfo on device
  magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );


  // arrays must be in host memory for magma svd
  A_p = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  U_p = ( double * ) malloc( m_A * m_A * sizeof( double ) );
  Vt_p = ( double * ) malloc( n_A * n_A * sizeof( double ) );
  ss_p = ( double * ) malloc( A_mn * sizeof( double ) );

  magma_dgetmatrix( m_A, n_A, A_pg, ldim_A, A_p, m_A );
  magma_dgetmatrix( m_A, m_A, U_pg, ldim_U, U_p, m_A );
  magma_dgetmatrix( n_A, n_A, Vt_pg, ldim_Vt, Vt_p, n_A );

  // compute size of work array
  magma_dgesdd( MagmaAllVec,
                m_A, n_A, A_p, m_A,
                ss_p,
                U_p, m_A,
                Vt_p, n_A,
                work_h, -1, iwork_h, magInfo );

  lwork = work_h[ 0 ];

  // compute factorization  
  magma_dgesdd( MagmaAllVec,
                m_A, n_A, A_p, m_A,
                ss_p,
                U_p, m_A,
                Vt_p, n_A,
                work_h, lwork, iwork_h, magInfo );
  assert( * magInfo == 0 );

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
  int A_mn;

  // get max and min
  A_mn = min( m_A, n_A );

  // allocate space for devInfo on device
  magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );

  // arrays must be in host memory for magma svd
  A_p = ( double * ) malloc( m_A * n_A * sizeof( double ) );
  U_p = ( double * ) malloc( m_A * A_mn * sizeof( double ) );
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
static void local_left_svecs_v2( cublasHandle_t cublasH,
				int m_A, int n_A, double * A_pg, int ldim_A,
				double * tau_h,
				double * work_h, int * iwork_h,
				double * T_d, int ldim_T,
				double * TVt_d, int ldim_TVt ) {
  // given an m_A x n_A matrix A stored in device memory in A_pg,
  // this function returns the first min(m_A,n_A) left singular
  // vectors in the first min(m_A,n_A) columns of A

  // this function requires that m_A >= n_A

  // declare and initialize auxiliary variables
  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0;

  double * ones_vc_d, * zeros_vc_d;

  double * ss_p;
  double * R_p, * U_p, * Vt_p;
  double * U_pg, * R_pg;
  double * work_d;
  int ldim_U, ldim_R, ldim_work;
  int i,j;

  int lwork = 0; // size of work buffer
  magma_int_t * magInfo = NULL; // stored on host 

  // vars for determining size of work array
  int nb = magma_get_dgesvd_nb( m_A, n_A );
  int A_mn;

  // get max and min
  A_mn = min( m_A, n_A );

  // some more initializations
  ldim_U = A_mn;
  ldim_R = A_mn;

  // allocate space for magInfo
  magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );
  
  // allocate space for auxiliary arrays and initialize
  cudaMalloc( ( void ** ) & ones_vc_d, n_A * sizeof( double ) );
  cudaMalloc( ( void ** ) & zeros_vc_d, n_A * sizeof( double ) );
  
  set_vc_one( n_A, ones_vc_d );
  set_vc_zero( n_A, zeros_vc_d );

  // allocate space for intermediate matrix on device 
  cudaError_t cudaStat;
  cudaStat = cudaMalloc( ( void ** ) & U_pg, sizeof( double ) * A_mn * A_mn );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( ( void ** ) & R_pg, sizeof( double ) * A_mn * A_mn );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( ( void ** ) & work_d, sizeof( double ) * A_mn * A_mn );
  assert( cudaStat == cudaSuccess );
  

  // arrays must be in host memory for magma svd
  R_p = ( double * ) malloc( A_mn * A_mn * sizeof( double ) );
  Vt_p = ( double * ) malloc( A_mn * A_mn * sizeof( double ) );
  U_p = ( double * ) malloc( m_A * A_mn * sizeof( double ) );
  ss_p = ( double * ) malloc( A_mn * sizeof( double ) );

  // if m_A <= n_A, just use the function that works for all cases
  // (matrix will never be very big in this case for this algorithm)
  if ( m_A <= n_A ) {

	local_left_svecs( m_A, n_A, A_pg, ldim_A,
			work_h, iwork_h );
    
	// free memory 
	free( Vt_p ); 
	free( U_p ); 
	free( ss_p );
	free( R_p );

	free( magInfo );
	
	cudaFree( U_pg );
	cudaFree( R_pg );
    cudaFree( work_d );

	cudaFree( ones_vc_d );
	cudaFree( zeros_vc_d );
    
	return;

  }

  // compute QR of A
  local_magqr_nopiv( m_A, n_A, 
			A_pg, ldim_A,
			tau_h, magInfo );

  // extract upper triangular matrix R
  
    // matrix needs to be in host memory for magma svd
    magma_dgetmatrix( A_mn, A_mn, A_pg, ldim_A, R_p, A_mn );

    // make R upper triangular
    for ( i=0; i < A_mn; i++ ) {
	  for ( j=0; j < A_mn; j++ ) {
	    if ( i > j ) {
		  R_p[ i + j * A_mn ] = 0.0;
		}
	  }
	}
	// TODO: if we use this function, move this loop to an aux function

  // determine size of work array
  magma_dgesdd( MagmaOverwriteVec,
				A_mn, A_mn, R_p, A_mn,
				ss_p,
				NULL, A_mn,
				Vt_p, A_mn,
				work_h, -1, iwork_h, magInfo );

  lwork = work_h[ 0 ];

  // compute factorization
  magma_dgesdd( MagmaOverwriteVec,
                A_mn, A_mn, R_p, A_mn,
                ss_p,
                NULL, A_mn,
                Vt_p, A_mn,
				work_h, lwork, iwork_h, magInfo );

  // transfer results to device
  magma_dsetmatrix( A_mn, A_mn, R_p, A_mn, U_pg, A_mn );

  // compute left svecs of A: Q(:,1:A_mn)*U

	magma_dlarft( cublasH, m_A, n_A, 
				A_pg, ldim_A,
				tau_h,
				T_d, ldim_T,
				TVt_d, ldim_TVt );

	// change A_pg into V needed for muliplication
	magma_dcopymatrix( n_A, n_A, A_pg, ldim_A, R_pg, ldim_R );

	Make_upper_tri( A_mn, A_mn, ldim_R, R_pg );
    // TODO: fix the order of this function's arguments

	mm_A_minus_B( A_mn, A_mn, A_pg, ldim_A, R_pg, ldim_R );
   
	replace_diag_vec( m_A, n_A, A_pg, ldim_A, ones_vc_d );
    	
	// begin multiplication
	
	ldim_work = A_mn;
	
	// work <-- TVt(:,1:p+b) * U
	cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
					A_mn, A_mn, A_mn,
					& d_one,
					TVt_d, ldim_TVt,
					U_pg, ldim_U,
					& d_zero,
					work_d, ldim_work );	

	// TVt_pg <-- - V * work 
	// TVt is just used as a placeholder here; that's why m_A
	// is used as the leading dim for the dgemm
	cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
					m_A, A_mn, A_mn, 
					& d_neg_one, 
					A_pg, ldim_A, 
					work_d, ldim_work, 
					& d_zero, 
					TVt_d, m_A );
	  
	// transfer over to A_pg: A_pg <-- TVt_pg
	magma_dcopymatrix( m_A, n_A, TVt_d, m_A,
				  A_pg, ldim_A );
	
	// finish calculation by computing A_pg <-- A_pg + [U; 0]
    mm_A_plus_B( A_mn, A_mn,
			A_pg, ldim_A,
			U_pg, ldim_U );
     

  // free memory
  free( Vt_p );
  free( U_p );
  free( ss_p );
  free( R_p );

  free( magInfo );
  
  cudaFree( U_pg );
  cudaFree( R_pg );
  cudaFree( work_d );

  cudaFree( ones_vc_d );
  cudaFree( zeros_vc_d );
}

// ========================================================================
static void project_away( cublasHandle_t cublasH,
				int m_A, int n_A, double * A_d, int ldim_A,
				int m_V, int n_V, double * V_d, int ldim_V ) {
  // given an m x n1 matrix A and an m x n2 matrix V, where V has ON cols,
  // this function projects the cols of A away from the subspace spanned
  // by the cols of V
  // i.e. it computes A - V*V'*A
  // A and V are both stored on the device
  // A is overwritten with the results of the computation

  // initialize aux vars
  double * work_d;
  int ldim_work;

  ldim_work = n_V;

  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0;

  // allocate memory for work array
  cudaError_t cudaStat;
  cudaStat = cudaMalloc( ( void ** ) & work_d, sizeof( double ) * n_V * n_A );
  assert( cudaStat == cudaSuccess );

  // perform projection; requires two dgemms

    // work <-- V'*A
	cublasDgemm( cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
					n_V, n_A, m_A, 
					& d_one, 
					V_d, ldim_V, 
					A_d, ldim_A, 
					& d_zero, 
					work_d, ldim_work );
     

	// A <-- A - V*work 
	cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
					m_V, n_A, n_V, 
					& d_neg_one, 
					V_d, ldim_V, 
					work_d, ldim_work, 
					& d_one, 
					A_d, ldim_A );

  // free memory for work array
  cudaFree( work_d );

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
static void my_dormqr_gpu( cublasHandle_t handle,
				char side, char trans, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				int m_B, int n_B, double * B_pg, int ldim_B,
				double * TVt_pg, int ldim_TVt ) {
  // applies the HH matrix H = I - VTV' to B
  // V is the lower triangular matrix with ones on the diagonal
  // and the HH reflectors stored in the columns
  // A_pg: matrix A after a call to dgeqrf; it holds the HH reflectors
  //	   in its lower triangular part

  // auxiliary vars
  int ldim_work;
  double * R_pg;
  double * work_d;
  int ldim_R = n_A;
  int k = n_A; // this is the number of HH reflectors 
  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0;

  double * ones_vc_d, * zeros_vc_d;

  cudaMalloc( ( void ** ) & R_pg, n_A * n_A * sizeof( double ) );
  cudaMalloc( ( void ** ) & work_d, k * max( m_B, n_B ) * sizeof( double ) );
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
	cublasDgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N,
					k, n_B, m_B,
					& d_one,
					A_pg, ldim_A,
					B_pg, ldim_B,
					& d_zero,
					work_d, ldim_work );	

	// B <-- - TVt' * work + B;
    cublasDgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, 
					m_B, n_B, k, 
					& d_neg_one, 
					TVt_pg, ldim_TVt, 
					work_d, ldim_work, 
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
					work_d, ldim_work );

    // B <-- - work * TVt + B
    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, 
					m_B, n_B, k, 
					& d_neg_one, 
					work_d, ldim_work,
					TVt_pg, ldim_TVt,
					& d_one,
					B_pg, ldim_B );
  
  }

  // restore the upper diagonal part of A
  replace_diag_vec( m_A, n_A, A_pg, ldim_A, zeros_vc_d );

  mm_A_plus_B( n_A, n_A, A_pg, ldim_A, R_pg, ldim_R );

  // free memory
  cudaFree( R_pg );
  cudaFree( work_d );
  cudaFree( ones_vc_d );
  cudaFree( zeros_vc_d );

}

// =======================================================================
static void oversample_downdate_gpu( cublasHandle_t handle,
				int m_Vloc, int n_Vloc, double * Vloc_pg, int ldim_Vloc,
				int m_Y_os, int n_Y_os, double * Y_os_pg, int ldim_Y_os,
				double * TVt_pg, int ldim_TVt ) {
  // downdates the oversamples in Y (Ytmp) to use in the next iteration
  // if H = I - VTV' was the transform computed using Y, then this function
  // computes H(:,b+1:end)'*Ytmp;
  //
  // Vloc holds the HH reflectors in its lower triangular part at the start;
  // Vloc holds the downdated samples at the end
  // Y_os holds the oversamples to be downdated

  // auxiliary vars
  int ldim_work;
  double * R_pg, * work_d;
  int ldim_R = n_Vloc;
  int k = n_Vloc; // this is the number of HH reflectors 
  double d_one = 1.0, d_zero = 0.0, d_neg_one = -1.0;

  double * ones_vc_d, * zeros_vc_d;

  cudaMalloc( ( void ** ) & R_pg, n_Vloc * n_Vloc * sizeof( double ) );
  cudaMalloc( ( void ** ) & work_d, k * n_Y_os * sizeof( double ) );
  cudaMalloc( ( void ** ) & ones_vc_d, n_Vloc * sizeof( double ) );
  cudaMalloc( ( void ** ) & zeros_vc_d, n_Vloc * sizeof( double ) );

  // initialize vectors of ones and zeros
  set_vc_one( n_Vloc, ones_vc_d );
  set_vc_zero( n_Vloc, zeros_vc_d );

  // quick exit if possible
  if ( m_Y_os == 0 || n_Y_os == 0 ) {
    return;
  }

  // change A_pg into V needed for mult; store upper triangular part in
  // R so we can restore it later

  magma_dcopymatrix( n_Vloc, n_Vloc, Vloc_pg, ldim_Vloc, R_pg, ldim_R );

  Make_upper_tri( n_Vloc, n_Vloc, ldim_R, R_pg );

  mm_A_minus_B( n_Vloc, n_Vloc, Vloc_pg, ldim_Vloc, R_pg, ldim_R );
 
  replace_diag_vec( m_Vloc, n_Vloc, Vloc_pg, ldim_Vloc, ones_vc_d );

  // begin multiplication

  // compute H(:,b+1:end)' * Y_os
  
  ldim_work = k;

  // work <-- V' * Y_os
  cublasDgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N,
				  k, n_Y_os, m_Y_os,
				  & d_one,
				  Vloc_pg, ldim_Vloc,
				  Y_os_pg, ldim_Y_os,
				  & d_zero,
				  work_d, ldim_work );	

  // Y_os(b+1:end,:) <-- - TVt(:,b+1:end)' * work + Y_os(b+1:end,:);
  // Note that Vloc holds the relevant output for the function
  cublasDgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, 
				  m_Y_os-k, n_Y_os, k, 
				  & d_neg_one, 
				  & TVt_pg[0+k*ldim_TVt], ldim_TVt, 
				  work_d, ldim_work, 
				  & d_one, 
				  & Y_os_pg[k+0*ldim_Y_os], ldim_Y_os );

  // store downdated samples in Vloc, which also serves as the sampling matrix for next iteration
  magma_dcopymatrix( m_Y_os-k, n_Y_os, & Y_os_pg[k+0*ldim_Y_os], ldim_Y_os,
				& Vloc_pg[k+0*ldim_Vloc], ldim_Vloc );

  // free memory
  cudaFree( R_pg );
  cudaFree( work_d );
  cudaFree( ones_vc_d );
  cudaFree( zeros_vc_d );

  // TODO: don't need R matrix in this function

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

// ======================================================================== 
static timespec start_timer( void ) { 
  // this function returns a timespec object that contains
  // clock information at the time of this function's execution
  //
  // performs the same function as MATLAB's 'tic'
 
  // declare variables
  timespec t1;

  // get current clock info
  cudaDeviceSynchronize();
  clock_gettime( CLOCK_MONOTONIC, & t1 );

  return t1;

}
	
// ======================================================================== 
static double stop_timer( timespec t1 ) {
  // this function returns a variable of type double that
  // corresponds to the number of seconds that have elapsed
  // since the time that t1 was generated by start_timer
  // 
  // performs the same function as MATLAB's 'toc'
  //
  // t1: the output of start_timer; holds clock information
  //     from a function call to start_timer
  
  // declare variables 
  timespec  t2;
  uint64_t  t_elapsed_nsec;
  double    t_elapsed_sec;

  // get current clock info
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, & t2);

  // calculate elapsed time
  t_elapsed_nsec = (1000000000L) * (t2.tv_sec - t1.tv_sec) + t2.tv_nsec - t1.tv_nsec;
  t_elapsed_sec = (double) t_elapsed_nsec / (1000000000L);

  return t_elapsed_sec;

}
