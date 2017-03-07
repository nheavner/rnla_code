#include <math.h>
#include <stdlib.h>
#include<stdio.h>
#include<stdint.h>
#include<mkl.h>

#define l_max( a, b ) ( (a) >  (b) ? (a) : (b) )
#define l_min( a, b ) ( (a) < (b) ? (a) : (b) )


// ===================================================================
// Declaration of local prototypes

static void matrix_generate(int m_A, int n_A, double * buff_A, int ldim_A);

static void print_double_matrix(char * name, int m_A, int n_A,
		double * buff_A, int ldim_A);

// ===================================================================

int main( int argc, char *argv[] ) {
  MKL_INT	m_A, n_A, ldim_A;
  int   i, j;
  double * buff_A;

  double Anorm;
  char f = 'F';

  // create matrix A
  m_A = 2;
  n_A = 2;
  ldim_A = l_max(1,m_A);
  buff_A = ( double * ) malloc( m_A * n_A * sizeof(double) );
  //generate matrix A
  for (i = 0; i < m_A; i++) { for (j = 0; j < n_A; j++) { buff_A[i + n_A * j] = i+1; } }//matrix_generate(m_A,n_A,buff_A,ldim_A);

  print_double_matrix("A",m_A,n_A,buff_A,m_A);

  // || A ||
  Anorm = dlange_( & f, & m_A, & n_A,
				buff_A, & m_A, NULL );
  //print results
  printf("norm computation complete\n");
  printf("|| A || = %f \n", Anorm);

  // free matrices
  free(buff_A);

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
