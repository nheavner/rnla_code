#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define min( a,b ) ( (a) > (b) ? (b) : (a) )

int Set_to_identity( int m_A, int n_A, double * A_pc, int ldim_A ) {

  // This function sets contents of matrix A, stored in cpu,
  // to the identity matrix

  int i,j;
  int mn_A = min( m_A, n_A );

  // Set the full matrix to 0
  for ( j=0; j<n_A; j++ ) {
    for ( i=0; i<m_A; i++ ) {
	  A_pg[ i + j * ldim_A ] = 0.0; 
	}
  }

  // Set the main diagonal to 1
  for ( i=0; i < mn_A; i++ ) {
    A_pg[ i + i * ldim_A ] = 1.0;
  }

  return 0;

}


void print_double_matrix( const char * name, int m_A, int n_A, 
                double * buff_A, int ldim_A ) {
  int  i, j;

  printf( "%s = [\n", name );
  for( i = 0; i < m_A; i++ ) {
    for( j = 0; j < n_A; j++ ) {
      printf( "%le ", buff_A[ i + j * ldim_A ] );
    }
    printf( "\n" );
  }
  printf( "];\n" );
}

int main() {

  // declare, initialize variables
  int m_A, n_A, ldim_A;
  m_A = 5; n_A = 5; ldim_A = m_A;
  const char * A_name = "A";
  double * A_p, * A_pg;

  // allocate array on host (cpu)
  A_p = ( double * ) malloc( m_A * n_A * sizeof( double ) );

  // call function
  Set_to_identity( m_A, n_A, A_p, ldim_A );

  // print out result
  print_double_matrix( A_name, m_A, n_A, A_p, ldim_A );

  // free memory
  free( A_p );

  return 0;
}
