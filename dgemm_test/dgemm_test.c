#include <math.h>
#include <stdlib.h>
#include<stdio.h>
#include<stdint.h>

#define l_max( a, b ) ( (a) >  (b) ? (a) : (b) )
#define l_min( a, b ) ( (a) < (b) ? (a) : (b) )


// ===================================================================
// Declaration of local prototypes

static void matrix_generate(int m_A, int n_A, double * buff_A, int ldim_A);

static void print_double_matrix(char * name, int m_A, int n_A,
		double * buff_A, int ldim_A);

// ===================================================================

int main( int argc, char *argv[] ) {
  int64_t	nb_alg, pp, m_A, n_A, ldim_A, m_B, n_B, ldim_B, ldim_C;
  int   i, j;
  double * buff_A, * buff_B, * buff_C;
  char all = 'A', t = 'T', n = 'N';
  double d_one = 1.0, d_zero = 0.0;

  // create matrices A, B, C
  m_A = 3;
  n_A = 3;
  ldim_A = l_max(1,m_A);
  m_B = 3;
  n_B = 3;
  ldim_B = l_max(1,m_B);
  ldim_C = ldim_A;

  buff_A = ( double * ) malloc( m_A * n_A * sizeof(double) );
  buff_B = ( double * ) malloc( m_B * n_B * sizeof(double) );
  buff_C = ( double * ) malloc( m_A * n_A * sizeof(double) );

  //generate matrices A,B
  for (i = 0; i < m_A; i++) { for (j = 0; j < n_A; j++) { buff_A[i + n_A * j] = i+1; } }//matrix_generate(m_A,n_A,buff_A,ldim_A);
  for (i = 0; i < m_B; i++) { for (j = 0; j < n_B; j++) { buff_B[i + n_B * j] = i + 1; } }//matrix_generate(m_B,n_B,buff_B,ldim_B);

  print_double_matrix("A",m_A,n_A,buff_A,m_A);
  print_double_matrix("B",m_B,n_B,buff_B,m_B);

  //compute C = A * B
  dgemm_( & n, & n, & m_A, & n_B, & n_A, 
	  & d_one, buff_A, & ldim_A,
	  	   buff_B, & ldim_B,
	  & d_zero,buff_C, & ldim_C);
  //print results
  printf("Multiplication complete\n");
  print_double_matrix("C",m_A,n_B,buff_C,m_A);

  // free matrices
  free(buff_A);
  free(buff_B);
  free(buff_C);

  printf("%% End of Program\n");

  return 0;
}

// ===============================================================================
static void matrix_generate(int m_A, int n_A, double * buff_A, int ldim_A) {
  int i,j;

  srand(10);
  for (j = 0; j < n_A; j++) {
    for (i=0; i < m_A; i++) {
      buff_A[i + j * ldim_A] = (double) rand() / (double) RAND_MAX;
    }
  }
}

// ===============================================================================
static void print_double_matrix(char * name, int m_A, int n_A, double * buff_A, int ldim_A) {
  int i,j;

  printf( "%s = [\n",name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%le ", buff_A[ i + j * ldim_A ] );
    }
    printf( "\n" );
  }
  printf( "];\n" );
}
