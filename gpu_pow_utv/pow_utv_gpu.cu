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
    printf( ";\n" );
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
    printf( ";\n" );
  }
  printf( "];\n" );

}
// ========================================================================
// Declaration of local prototypes

static void Normal_random_matrix( int m_A, int n_A,
				double * A_pg, int ldim_A,
				curandGenerator_t rand_gen );

static void Set_to_identity( int m_A, int n_A, double * A_pc, int ldim_A );

static void gpu_dgemm( cublasHandle_t handle,
				char opA, char opB, double alpha, 
				int m_A, int n_A, double * A_pg, int ldim_A,
				int m_B, int n_B, double * B_pg, int ldim_B,
				double beta,
				int m_C, int n_C, double * C_pg, int ldim_C );

static timespec start_timer( void );

static double stop_timer( timespec t1 );
// ========================================================================

// Main function
int pow_utv_gpu( 
		int m_A, int n_A, double * A_pc, int ldim_A,
		int build_U, int m_U, int n_U, double * U_pc, int ldim_U,
		int build_V, int m_V, int n_V, double * V_pc, int ldim_V,
		int q_iter ) {

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
  char t = 'T', n = 'N';
  int i, j;
  int m_G, n_G, ldim_G,
	  m_Y, n_Y, ldim_Y,
	  m_R, n_R, ldim_R,
	  m_R_block, n_R_block, ldim_R_block;
  double * A_pg;  // pointer to matrix array in gpu
  double * G_pg, * Y_pg, * R_pg, * R_block_pg, * eye_pg;
  double * eye_pc;
  double * tau_pg;
 
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
  // TODO: do we still need all these for powerURV?
  magma_int_t * magInfo;
  double * T_d; // T matrix for applying HH matrices
  double * tau_h; // tau vector for applying HH matrices

#ifdef PROFILE
  double tt_spl,
		 tt_qr1_fact, tt_qr1_updt_a,
		 tt_qr2_fact;
  
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
  tt_qr1_updt_a  = 0.0;
  tt_qr2_fact    = 0.0;
#endif

  // initialize auxiliary variables
  m_G = n_A;	n_G	= n_A;	ldim_G = n_A;
  m_Y = m_A;	n_Y	= n_A;	ldim_Y = m_A;
  m_R = m_A;    n_R = n_A;  ldim_R = m_R;

  m_R_block = magma_get_dgeqrf_nb( m_A, n_A ); 
  n_R_block = min( m_A, n_A ); 
  ldim_R_block = m_R_block;
 
  // initialize auxiliary objects
  cudaError_t cudaStat = cudaSuccess; 
  cudaStat = cudaMalloc( & A_pg, m_A * n_A * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & G_pg, m_G * n_G * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & Y_pg, m_Y * n_Y * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & R_pg, m_R * n_R * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  cudaStat = cudaMalloc( & R_block_pg, m_R_block * n_R_block * sizeof( double ) );
  assert( cudaStat == cudaSuccess );

  magInfo = ( magma_int_t * ) malloc( sizeof( magma_int_t ) );
  cudaMalloc( ( void ** ) & devInfo, sizeof( int ) );
  
  // determine max size of work arrays for magma qr calcs
  // and allocate memory for the arrays
  int nb_mn, nb_nn, T_len_mn, T_len_nn;
  nb_mn = magma_get_dgeqrf_nb( m_A, n_A );
  nb_nn = magma_get_dgeqrf_nb( n_A, n_A );
  T_len_mn = ( 2 * min( m_A, n_A ) + ceil( n_A / 32.0 ) * 32 ) * nb_mn;
  T_len_nn = ( 2 * n_A + ceil( n_A / 32.0 ) * 32 ) * nb_nn;
  cudaMalloc( ( void ** ) & T_d, 
			  max( T_len_mn, T_len_nn ) * sizeof( double ) );
  tau_h = ( double * ) malloc( max( m_A, n_A ) * sizeof( double ) );

  // create identity matrix necessary for inverting stupid T_d blocks and store in the gpu
  eye_pc = ( double * ) malloc( nb_mn * nb_mn * sizeof( double ) );
  cudaStat = cudaMalloc( & eye_pg, nb_mn * nb_mn * sizeof( double ) );
  assert( cudaStat == cudaSuccess );
  Set_to_identity( nb_mn, nb_mn, eye_pc, nb_mn );
  cudaMemcpy( eye_pg, eye_pc, nb_mn * nb_mn * sizeof( double ),
		      cudaMemcpyHostToDevice );

  // copy A to gpu
  cudaMemcpy( A_pg, A_pc, m_A * n_A * sizeof( double ),
		      cudaMemcpyHostToDevice );

  // Compute the "sampling" matrix Y
  // Aloc = T([J2,I3],[J2,I3]);
  // Y = Aloc' * randn(m-(i-1)*b,b+p);
#ifdef PROFILE
  time1 = start_timer();
#endif
  // Create random matrix for sampling
  Normal_random_matrix( m_G, n_G, G_pg, ldim_G,
						rand_gen );

  // carry out "sampling" multiplications
  // for i_iter = 1:q_iter:
  //   Y = Aloc' * (Aloc * Y);
  // end
  for( j=0; j<q_iter; j++ ) {

/*
	// Y <-- A*G
	gpu_dgemm( cublasH, 
			   n, n, d_one,
			   m_A, n_A, A_pg, ldim_A,
			   m_G, n_G, G_pg, ldim_G,
			   d_zero, 
			   m_Y, n_Y, Y_pg, ldim_Y );
	
	// orthonormalize columns of Y for stability
	magma_dgeqrf3_gpu( m_Y, n_Y, Y_pg, ldim_Y,
					  tau_h, T_d, magInfo );
	
	magma_dorgqr_gpu( m_Y, n_Y, min( m_Y, n_Y ), Y_pg, ldim_Y,
					  tau_h, T_d, nb_mn, magInfo );
	
	// complete iteration, reusing G for storage; G <-- A'*Y
	gpu_dgemm( cublasH, 
			   t, n, d_one,
			   m_A, n_A, A_pg, ldim_A,
			   m_Y, n_Y, Y_pg, ldim_Y,
			   d_zero,
			   m_G, n_G, G_pg, ldim_G );

	// orthonormalize columns of G for stability
	magma_dgeqrf3_gpu( m_G, n_G, G_pg, ldim_G,
					  tau_h, T_d, magInfo );

	magma_dorgqr_gpu( m_G, n_G, min( m_G, n_G ), G_pg, ldim_G,
					  tau_h, T_d, nb_nn, magInfo );
	*/

  }

#ifdef PROFILE
  tt_spl += stop_timer( time1 );
  time1 = start_timer();
#endif

  // Apply the transform V (stored in G) to the right of A; store result in Y; i.e. Y <- A*V
  /*
  gpu_dgemm( cublasH, 
			 n, n, d_one,
			 m_A, n_A, A_pg, ldim_A,
			 m_G, n_G, G_pg, ldim_G,
			 d_zero,
			 m_Y, n_Y, Y_pg, ldim_Y );
*/

#ifdef PROFILE
  tt_qr1_updt_a += stop_timer( time1 );
  time1 = start_timer();
#endif

  // TODO: the cudamemcpy functions should be changed to magma_copymatrix, at least when
  //       a submatrix is possibly being copied

  // %%% Next compute QR factorization of A*V to get A*V = (U*R)*V
  /*
  magma_dgeqrf3_gpu( m_Y, n_Y, Y_pg, ldim_Y,
					tau_h, T_d, magInfo );
  */

  // have to extract part of R from T_d because of the stupid MAGMA API
  cudaStat = cudaMemcpy( R_pg, Y_pg, 
					     m_Y * n_Y * sizeof( double ), cudaMemcpyDeviceToDevice );
  assert( cudaStat == cudaSuccess );

  cudaStat = cudaMemcpy( R_block_pg, & T_d[ min( m_A, n_A ) * nb_mn ], 
					     nb_mn * ( min( m_A, n_A ) ) * sizeof( double ), cudaMemcpyDeviceToDevice );
  assert( cudaStat == cudaSuccess );
/*
  // have to invert the blocks in T_d because of the stupid MAGMA API
  for ( j=0; j + nb_mn < min( m_A, n_A ); j+= nb_mn ) {
    // invert the stupid block
	magma_dtrsm( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
				 nb_mn, nb_mn, 1.0, 
				 & R_block_pg[ 0 + j * ldim_R_block ], ldim_R_block,
				 eye_pg, nb_mn );

	// transfer the stupid block over to R
	magma_dcopymatrix( nb_mn, nb_mn, eye_pg, nb_mn, & R_pg[ j + j * ldim_R ], ldim_R );

	// reset the identity matrix
	cudaStat = cudaMemcpy( eye_pg, eye_pc, 
						   nb_mn * nb_mn * sizeof( double ), cudaMemcpyHostToDevice );
	assert( cudaStat == cudaSuccess );
  }
 */ 

  // form the orthogonal matrix U
  // TODO: this assumes the matrix is square; code we release to the public shouldn't have
  //       this assumption, so fix it if you release it!!!!!!
  /*
  magma_dorgqr_gpu( m_Y, n_Y, min( m_Y, n_Y ), Y_pg, ldim_Y,
					tau_h, T_d, nb_mn, magInfo );
*/	
#ifdef PROFILE
	tt_qr2_fact += stop_timer( time1 );
	time1 = start_timer();
#endif

  // transfer U, R, V from device to host
  cudaStat = cudaMemcpy( A_pc, R_pg, m_A * n_A * sizeof( double ),
						 cudaMemcpyDeviceToHost );
  assert( cudaStat == cudaSuccess );

  //TODO: put this into a function
  
  for ( j=0; j < n_A; j++ ) {
    for ( i=0; i < m_A; i++ ) {
	  if ( i > j ) {
	    A_pc[ i + j * ldim_A ] = 0.0; // have to remove the HH information from the lower
									  // part of the R factor
	  }
	}
  }
  

  cudaStat = cudaMemcpy( U_pc, Y_pg, m_A * m_A * sizeof( double ),
						 cudaMemcpyDeviceToHost );
  assert( cudaStat == cudaSuccess );
  
  cudaStat = cudaMemcpy( V_pc, G_pg, n_A * n_A * sizeof( double ),
						 cudaMemcpyDeviceToHost );
  assert( cudaStat == cudaSuccess );
  
  // remove auxiliary objects
  cudaFree( A_pg );

  cudaFree( G_pg );
  cudaFree( Y_pg );
  cudaFree( R_pg );
  cudaFree( R_block_pg );

  free( magInfo );

  cudaFree( T_d );
  free( tau_h );

  free( eye_pc );
  cudaFree( eye_pg );

  // finalize magma
  magma_finalize();
  
  // destroy the handles for the blas and solver environments
  cublasDestroy( cublasH );
  cusolverDnDestroy( cusolverH );

  cudaFree( devInfo );

#ifdef PROFILE
  printf( "%% tt_build_y:	%le \n", tt_spl );
  printf( "%% tt_qr1_updt_a:	%le \n", tt_qr1_updt_a );
  printf( "%% tt_qr2_fact:	%le \n", tt_qr2_fact );
  printf( "%% total_time:	%le \n", 
		   tt_spl + 
		   tt_qr1_updt_a + 
		   tt_qr2_fact );
#endif


  return 0;

}

// ========================================================================
// Auxiliary functions 

static void Normal_random_matrix( int m_A, int n_A,
				double * A_pg, int ldim_A,
				curandGenerator_t rand_gen ) {
  // fills the gpu array with random numbers
  // of standard normal distribution
  curandGenerateNormalDouble( rand_gen, A_pg, m_A * n_A,
								0.0, 1.0 );
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
