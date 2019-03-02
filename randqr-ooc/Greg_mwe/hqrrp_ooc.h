int hqrrp_ooc_multithreaded( char * dir_name, size_t dir_name_size, 
		char * A_fname, size_t A_fname_size,
		int m_A, int n_A, int ldim_A,
        int * buff_jpvt, double * buff_tau,
        int nb_alg, int kk, int pp, int panel_pivoting );

int hqrrp_ooc_physical_pivot( char * dir_name, size_t dir_name_size, 
		char * A_fname, size_t A_fname_size,
		int m_A, int n_A, int ldim_A,
        int * buff_jpvt, double * buff_tau,
        int nb_alg, int kk, int pp, int panel_pivoting );
