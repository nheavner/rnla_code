n = [500,1000,2000,3000,4000,5000,6000,8000,10000]; 

nn_1c = [8.30e-02 2.40e-01 1.36e+00 4.35e+00 9.82e+00 1.88e+01 3.23e+01 7.49e+01 1.45e+02]; 


nn_4c =  [6.18e-02 1.37e-01 5.95e-01 1.67e+00 3.52e+00 6.42e+00 1.08e+01 2.40e+01 4.55e+01]; 

noort_1c= [6.20e-02 2.42e-01 1.61e+00 5.15e+00 1.17e+01 2.25e+01 3.84e+01 9.00e+01 1.73e+02]; 

noort_4c = [3.81e-02 1.32e-01 6.73e-01 1.92e+00 4.10e+00 7.55e+00 1.27e+01 2.86e+01 5.44e+01]; 

rand_utv_1c = [1.34e-01 4.63e-01 3.11e+00 1.00e+01 2.29e+01 4.45e+01 7.60e+01 1.80e+02 3.45e+02]; 

rand_utv_4c = [9.06e-02 2.21e-01 1.16e+00 3.44e+00 7.52e+00 1.40e+01 2.40e+01 5.44e+01 1.05e+02]; 

p1 = plot(n, rand_utv_1c ./ noort_1c, 'rd-',...
	n, rand_utv_4c ./ noort_4c, 'rd-',...
	n, rand_utv_1c ./ nn_1c, 'bs-',...
	n, rand_utv_4c ./ nn_4c, 'bs-');

set(gca,'fontsize',12);
set(p1,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Speedup');
legend('t_{rutv}/t_{noort}, 1 core',...
	't_{rutv}/t_{noort}, 4 cores',...
	't_{rutv}/t_{nn}, 1 core',...
	't_{rutv}/t_{nn}, 4 cores');
legend('Location','SouthEast');
