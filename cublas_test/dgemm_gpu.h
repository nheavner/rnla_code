void dgemm_gpu ( char transA, char transB, int m, int n, int k,
				double * alpha, double * A_pg, int ldim_A,
				double * B_pg, int ldim_B,
				double * beta, double * C_pg, int ldim_C ); 

