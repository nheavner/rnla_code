clear


%%%%%%%%%%%%%%%%%%%%%%%%%%% read in data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
times_cpu
times_gpu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
p1 = plot(n_rutv_gpu, t_dgesdd_gpu ./ (n_rutv_gpu.^3), 'ko-',...
		n_rutv_gpu, t_dgeqp3_gpu ./ (n_rutv_gpu.^3), 'rx-',...
		n_rutv_gpu, t_randutv_gpu ./ (n_rutv_gpu.^3), 'bv-',...
		n_rutv_gpu, t_dgeqrf_gpu ./ (n_rutv_gpu.^3), 'gs-',...
		n_rutv_gpu, t_dgemm_gpu ./ (n_rutv_gpu.^3), 'md-');
ax = gca;
set(ax,'fontsize',16);
set(p1,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Time [s] / n^3');
title('GPU computations');
legend(	'MAGMA dgesdd (SVD)',...
		'MAGMA dgeqp3 (CPQR)',...
		'rand\_utv\_gpu, q=2',...
		'MAGMA dgeqrf (QR)',...
		'MAGMA dgemm (multiplication)');
legend('Location', 'NorthEast');

figure(2);
p2 = plot(n_rutv_cpu, t_dgesdd_cpu ./ (n_rutv_cpu.^3), 'ko-',...
		n_rutv_cpu, t_dgeqp3_cpu ./ (n_rutv_cpu.^3), 'rx-',...
		n_rutv_cpu, t_randutv_cpu ./ (n_rutv_cpu.^3), 'bv-',...
		n_rutv_cpu, t_dgeqrf_cpu ./ (n_rutv_cpu.^3), 'gs-',...
		n_rutv_cpu, t_dgemm_cpu ./ (n_rutv_cpu.^3), 'md-');
ax = gca;
set(ax,'fontsize',16);
set(p2,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Time [s] / n^3');
title('CPU computations');
legend(	'dgesdd (SVD)',...
		'dgeqp3 (CPQR)',...
		'rand\_utv, q=2',...
		'dgeqrf (QR)',...
		'dgemm (multiplication)');
legend('Location', 'NorthEast');

figure(3);
p3 = plot(n_rutv_cpu, t_dgesdd_cpu ./ t_dgesdd_gpu, 'ko-',...
		n_rutv_cpu, t_dgeqp3_cpu ./ t_dgeqp3_gpu, 'rx-',...
		n_rutv_cpu, t_randutv_cpu ./ t_randutv_gpu, 'bv-',...
		n_rutv_cpu, t_dgeqrf_cpu ./ t_dgeqrf_gpu, 'gs-',...
		n_rutv_cpu, t_dgemm_cpu ./ t_dgemm_gpu, 'md-');
ax = gca;
set(ax,'fontsize',16);
set(p3,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Time [s] / n^3');
title('CPU computations');
legend(	'dgesdd (SVD)',...
		'dgeqp3 (CPQR)',...
		'rand\_utv, q=2',...
		'dgeqrf (QR)',...
		'dgemm (multiplication)');
legend('Location', 'NorthEast');