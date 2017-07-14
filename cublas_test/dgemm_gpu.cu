#include <cublas_v2.h>

void dgemm_gpu ( char transA, char transB, int m, int n, int k,
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

