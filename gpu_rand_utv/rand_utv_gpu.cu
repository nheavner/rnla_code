#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

// =======================================================================
// Definition of macros

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) > (b) ? (b) : (a) )
#define dabs( a, b ) ( (a) >= 0.0 ? (a) : -(a) )

// =======================================================================
// Definition of global variables

static int gpu_thread_size = 256; // TODO: temporary value; determine how to optimize
						 // this gives the number of threads per block for
						 // kernel calls

// ========================================================================
// Declaration of local prototypes

static void Set_to_identity( int m_A, int n_A, double * A_pc, int ldim_A );

__global__ 
static void Normal_random_matrix_kern( int m_A, int n_A,
				double * A_pg, int ldim_A,
				curandState_t * states, 
				unsigned long long rand_seed );

static void Normal_random_matrix( int m_A, int n_A,
				double * A_pg, int ldim_A,
				curandState_t * states, 
				unsigned long long * rs_pt );

static void dgemm_gpu( char transA, char transB, int m, int n, int k,
				double * alpha, double * A_pg, int ldim_A,
				double * B_pg, int ldim_B,
				double * beta, double * C_pg, int ldim_C );

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

  // Declaration of variables
  double d_one = 1.0;
  double d_zero = 0.0;
  char t = 'T', n = 'N';
  int i, j;
  int mn_A;
  int m_G, n_G, ldim_G, m_G_loc, n_G_loc,
	  m_Y, n_Y, ldim_Y, m_Y_loc, n_Y_loc;
  int m_A_loc, n_A_loc; // this block is comprised of sections 22,23,32,33
  double * A_pg, * U_pg, * V_pg; // pointers to matrix arrays in gpu
  double * G_pg, * G_loc_pg, * A_loc_pg, * Y_pg, * Y_loc_pg;
  
  unsigned long long rand_seed = 7;
  unsigned long long * rs_pt = & rand_seed; 
  curandState_t * states; // we store a random state for every thread
 
  

 

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

  // initialize auxiliary variables
  mn_A = min( m_A, n_A );
  m_A_loc = m_A; n_A_loc = n_A;
  m_G = m_A; n_G = bl_size + pp; ldim_G = m_A;
  m_Y = n_A; n_Y = bl_size + pp; ldim_Y = n_A;
  

  // initialize auxiliary objects
  cudaMalloc( & A_pg, m_A * n_A * sizeof( double ) );
  cudaMalloc( & U_pg, m_U * n_U * sizeof( double ) );
  cudaMalloc( & V_pg, m_V * n_V * sizeof( double ) );
  
  cudaMalloc( & G_pg, m_G * n_G * sizeof( double ) );
  cudaMalloc( & Y_pg, m_Y * n_Y * sizeof( double ) );

  cudaMalloc( ( void ** ) & states, m_G * n_G * sizeof( curandState_t ) );


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
	m_A_loc = m_A - i;
	n_A_loc = n_A - i;
	m_G_loc = m_A - i;
	n_G_loc = bl_size + pp;
    m_Y_loc = n_A - i;
	n_Y_loc = bl_size + pp;

    
	A_loc_pg = & A_pg[ i + i * ldim_A ];
	G_loc_pg = & G_pg[ i + 0 * ldim_G ];
    Y_loc_pg = & Y_pg[ i + 0 * ldim_Y ]; 

	// Compute the "sampling" matrix Y
	// Aloc = T([J2,I3],[J2,I3]);
	// Y = Aloc' * randn(m-(i-1)*b,b+p);

      // Create random matrix for sampling
      Normal_random_matrix( m_G, n_G, G_pg, ldim_G, states, rs_pt );

      // carry out "sampling" multiplication
      dgemm_gpu( t, n, m_Y, n_Y, m_G,
				& d_one, A_pg, ldim_A,
				G_pg, ldim_G,
				& d_zero, Y_pg, ldim_Y );

	// perform "power iteration" if requested
	// for i_iter = 1:q_iter:
	//   Y = Aloc' * (Aloc * Y);
	// end
    for( j=0; j<q_iter; j++ ) {
	  // reuse G_loc for storage; G <-- A*Y
	  dgemm_gpu( n, n, m_G_loc, n_G_loc, m_Y_loc,
					& d_one, A_loc_pg, ldim_A,
					Y_loc_pg, ldim_Y,
					& d_zero, G_loc_pg, ldim_G );
	  
	  // complete iteration; Y <-- A'*G
	  dgemm_gpu( t, n, m_Y_loc, n_Y_loc, m_G_loc,
					& d_one, A_loc_pg, ldim_A,
					G_loc_pg, ldim_G,
					& d_zero, Y_loc_pg, ldim_Y );
	}

    
    // Construct the local transform to be applied "from the left".
    // if (p > 0)
    //   [~,~,Jtmp] = qr(Y,0);
    //   [Vloc,~,~] = qr(Y(:,Jtmp(1:b)));
    // else
    //   [Vloc,~]   = LOCAL_nonpiv_QR(Y,b);
    // end

	  // TODO

	  // TODO: remove capability for oversampling; it's unecessary

    // Apply the pivot matrix to rotate maximal mass into the "J2" column
	// T(:,[J2,J3]) = T(:[J2,J3])*Vloc;

	  // TODO
  
    
    // Update matrix V with transformations from the first QR.

	  // TODO

	  
    // %%% Next determine the rotations to be applied "from the left".
    // [Uloc,Dloc]      = LOCAL_nonpiv_QR(T([J2,I3],J2));

	  // TODO

	// update rest of matrix A with transformations from the second QR

      // TODO
	
	// update matrix U with transformations from the second QR

	  // TODO

	
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

	  // TODO

	// end of main loop  
  }

  // the final, potentially abnormally-sized block is processed inside the
  // previous loop

  // transfer arrays from device to host

    // TODO

  // remove auxiliary objects
  cudaFree( A_pg );
  cudaFree( U_pg );
  cudaFree( V_pg );

  cudaFree( G_pg );
  cudaFree( Y_pg );
  cudaFree( states );

  return 0;

}

// ========================================================================
// Auxiliary functions 

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
__global__ 
static void Normal_random_matrix_kern( int m_A, int n_A, 
				double * A_pg, int ldim_A,
				curandState_t * states, 
				unsigned long long rand_seed ) {
  // fill matrix A with random numbers of standard normal distribution
 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
 
  //seed RNG
  curand_init( rand_seed, i, 0, & states[ i ] );

  if (i < m_A * n_A) {
    A_pg[ i ] = curand_normal( & states[ i ]);
  }

}

static void Normal_random_matrix( int m_A, int n_A,
				double * A_pg, int ldim_A,
				curandState_t * states, 
				unsigned long long * rs_pt ) {
  // host function which fills the gpu array with random numbers
  // of standard normal distribution
Normal_random_matrix_kern<<<( m_A * n_A / gpu_thread_size ) + 1, gpu_thread_size >>>( m_A, n_A, A_pg, ldim_A, states, * rs_pt );

  * rs_pt = * rs_pt + 1;

}

// ========================================================================
static void dgemm_gpu( char transA, char transB, int m, int n, int k,
				double * alpha, double * A_pg, int ldim_A,
				double * B_pg, int ldim_B,
				double * beta, double * C_pg, int ldim_C ) {

  // generate the correct transpose option identifier that CUBLAS accepts
  cublasOperation_t cutransA, cutransB;

  if ( transA == 'N' ) { cutransA = CUBLAS_OP_N; }
  else if ( transA == 'T' ) { cutransA = CUBLAS_OP_T; }

  if ( transB == 'N' ) { cutransB = CUBLAS_OP_N; }
  else if ( transB == 'T' ) { cutransB = CUBLAS_OP_T; }

  // create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate( & handle );

  // do the multiplication
  cublasDgemm( handle, cutransA, cutransB, m, n, k, alpha,
				A_pg, ldim_A, B_pg, ldim_B, 
				beta, C_pg, ldim_C );
  
  // destroy the handle
  cublasDestroy( handle );

}
