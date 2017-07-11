#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>


// =======================================================================
// Definition of macros

#define max( a, b ) ( (a) > (b) ? (a) : (b) )
#define min( a, b ) ( (a) > (b) ? (b) : (a) )
#define dabs( a, b ) ( (a) >= 0.0 ? (a) : -(a) )

// ========================================================================
// Declaration of local prototypes

// ========================================================================
// Main function
int rand_utv_gpu( 
		int m_A, int n_A, double * A_pg, int ldim_A,
		int build_U, int m_U, int n_U, double * U_pg, int ldim_U,
		int build_V, int m_V, int n_V, double * V_pg, int ldim_V,
		int bl_size, int pp, int q_iter ) {

// randUTV: It computes the (rank-revealing) UTV factorization of matrix A.
//
// Matrices A, U, V must be stored in column-order
//
// Arguments:
// ----------
// m_A:      Number of rows of matrix A.
// n_A:      Number of columns of matrix A.
// A_pg:   Address of data in matrix A in gpu. Matrix to be factorized.
// ldim_A:   Leading dimension of matrix A.
// build_U:  If build_u==1, matrix U is built.
// m_U:      Number of rows of matrix U.
// n_U:      Number of columns of matrix U.
// U_pg:   Address of data in matrix U in gpu.
// ldim_U:   Leading dimension of matrix U.
// build_V:  If build_v==1, matrix V is built.
// m_V:      Number of rows of matrix V.
// n_V:      Number of columns of matrix V.
// V_pg:   Address of data in matrix V in gpu.
// ldim_V:   Leading dimension of matrix V.
// bl_size:   Block size. Usual values for nb_alg are 32, 64, etc.
// pp:       Oversampling size. Usual values for pp are 5, 10, etc.
// q_iter:   Number of "power" iterations. Usual values are 2.

  // Declaration of variables
  int i, j;
  int mn_A;

  // set seed for random generator
  srand( 15 );

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

  // initialize auxiliary objects

  // initialize auxiliary variables
  mn_A = min( m_A, n_A );

  // initialize U and V to identity
  
    // TODO

  // Main loop
  for ( i=0; i < mn_A; i += bl_size ) {

    // some initializations for every iteration
	
	// Compute the "sampling" matrix Y
	// Aloc = T([J2,I3],[J2,I3]);
	// Y = Aloc' * randn(m-(i-1)*b,b+p);

      // TODO

	// perform "power iteration" if requested
	// for i_iter = 1:q_iter:
	//   Y = Aloc' * (Aloc * Y);
	// end

	  // TODO

    
    // Construct the local transform to be applied "from the left".
    // if (p > 0)
    //   [~,~,Jtmp] = qr(Y,0);
    //   [Vloc,~,~] = qr(Y(:,Jtmp(1:b)));
    // else
    //   [Vloc,~]   = LOCAL_nonpiv_QR(Y,b);
    // end

	  // TODO

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

  // remove auxiliary objects

  return 0;

}

// ========================================================================
// Auxiliary functions 

