clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = [500,1000,2000,3000,4000,5000,6000,8000,10000]; 

svd_1c = [6.41e-02 4.29e-01 6.79e+00 2.67e+01 6.54e+01 1.29e+02 2.23e+02 5.41e+02 1.04e+03]; 

svd_4c = [9.76e-02 3.69e-01 6.62e+00 2.67e+01 6.57e+01 1.30e+02 2.26e+02 5.45e+02 1.05e+03]; 

nn_1c_q0 = [ 8.30e-02 2.40e-01 1.36e+00 4.35e+00 9.82e+00 1.88e+01 3.23e+01 7.49e+01 1.45e+02]; 

nn_4c_q0 = [6.18e-02 1.37e-01 5.95e-01 1.67e+00 3.52e+00 6.42e+00 1.08e+01 2.40e+01 4.55e+01]; 

noort_1c_q0 =  [6.20e-02 2.42e-01 1.61e+00 5.15e+00 1.17e+01 2.25e+01 3.84e+01 9.00e+01 1.73e+02]; 

noort_4c_q0 = [3.81e-02 1.32e-01 6.73e-01 1.92e+00 4.10e+00 7.55e+00 1.27e+01 2.86e+01 5.44e+01]; 

rutv_1c_q0 = [1.34e-01 4.63e-01 3.11e+00 1.00e+01 2.29e+01 4.45e+01 7.60e+01 1.80e+02 3.45e+02]; 

rutv_4c_q0 = [9.06e-02 2.21e-01 1.16e+00 3.44e+00 7.52e+00 1.40e+01 2.40e+01 5.44e+01 1.05e+02]; 

nn_1c_q1 =  [9.70e-02 2.92e-01 1.82e+00 5.87e+00 1.35e+01 2.57e+01 4.44e+01 1.04e+02 2.00e+02]; 

nn_4c_q1 = [5.25e-02 1.40e-01 7.33e-01 2.13e+00 4.58e+00 8.48e+00 1.44e+01 3.25e+01 6.20e+01]; 

noort_1c_q1 = [7.72e-02 3.01e-01 2.08e+00 6.73e+00 1.55e+01 2.99e+01 5.10e+01 1.20e+02 2.32e+02]; 

noort_4c_q1 = [7.78e-02 1.67e-01 8.05e-01 2.36e+00 5.16e+00 9.58e+00 1.62e+01 3.69e+01 7.12e+01]; 

rutv_1c_q1 = [9.61e-02 5.03e-01 3.56e+00 1.16e+01 2.66e+01 5.14e+01 8.84e+01 2.08e+02 4.02e+02]; 

rutv_4c_q1 = [9.76e-02 2.34e-01 1.30e+00 3.90e+00 8.58e+00 1.61e+01 2.76e+01 6.29e+01 1.21e+02]; 
 
nn_1c_q2 = [1.06e-01 3.44e-01 2.28e+00 7.41e+00 1.71e+01 3.32e+01 5.69e+01 1.34e+02 2.60e+02]; 

nn_4c_q2 = [7.87e-02 1.74e-01 8.64e-01 2.59e+00 5.63e+00 1.05e+01 1.79e+01 4.08e+01 7.86e+01]; 

noort_1c_q2 = [1.10e-01 3.76e-01 2.56e+00 8.33e+00 1.93e+01 3.70e+01 6.34e+01 1.49e+02 2.89e+02]; 

noort_4c_q2 = [5.37e-02 1.70e-01 9.47e-01 2.81e+00 6.22e+00 1.16e+01 1.98e+01 4.51e+01 8.73e+01]; 

rutv_1c_q2 = [1.57e-01 5.72e-01 4.04e+00 1.32e+01 3.04e+01 5.83e+01 1.00e+02 2.37e+02 4.64e+02]; 

rutv_4c_q2 = [5.68e-02 2.39e-01 1.43e+00 4.36e+00 9.62e+00 1.81e+01 3.12e+01 7.10e+01 1.37e+02]; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1);
p1 = plot(n, noort_1c_q0 ./ nn_1c_q0, 'kd-',...
	n, noort_4c_q0 ./ nn_4c_q0,'ks-',...
	n, noort_1c_q0 ./ rutv_1c_q0,'bd-',...
	n, noort_4c_q0 ./ rutv_4c_q0, 'bs-',...
	n, noort_1c_q0 ./ svd_4c,'rd-',...
	n, noort_4c_q0 ./ svd_4c,'rs-');
ax = gca;
set(ax,'fontsize',12);
set(p1,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Speedup');
legend('t_{noort}/t_{nn}, 1 core',...
	't_{noort}/t_{nn}, 4 cores',...
	't_{noort}/t_{rutv}, 1 core',...
	't_{noort}/t_{rutv}, 4 cores',...
	't_{noort}/t_{svd}, 1 core',...
	't_{noort}/t_{svd}, 4 cores');
legend('Location','SouthEast');
title('q = 0');
yl = ylim; % get limits on y axis for later
ax.YGrid = 'on';

%%%%%%%%%%
figure(2);
p2 = plot(n, noort_1c_q1 ./ nn_1c_q1, 'kd-',...
	n, noort_4c_q1 ./ nn_4c_q1,'ks-',...
	n, noort_1c_q1 ./ rutv_1c_q1,'bd-',...
	n, noort_4c_q1 ./ rutv_4c_q1, 'bs-',...
	n, noort_1c_q1 ./ svd_4c,'rd-',...
	n, noort_4c_q1 ./ svd_4c,'rs-');
ax = gca;
set(ax,'fontsize',12);
set(p2,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Speedup');
title('q = 1');
ylim(yl);
ax.YGrid = 'on';

%%%%%%%%%%
figure(3);
p3 = plot(n, noort_1c_q2 ./ nn_1c_q2, 'kd-',...
	n, noort_4c_q2 ./ nn_4c_q2,'ks-',...
	n, noort_1c_q2 ./ rutv_1c_q2,'bd-',...
	n, noort_4c_q2 ./ rutv_4c_q2, 'bs-',...
	n, noort_1c_q2 ./ svd_4c,'rd-',...
	n, noort_4c_q2 ./ svd_4c,'rs-');
ax = gca;
set(ax,'fontsize',12);
set(p3,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Speedup');
title('q = 2');
ylim(yl);
ax.YGrid = 'on';
