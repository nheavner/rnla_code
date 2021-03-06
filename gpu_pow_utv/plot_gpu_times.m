clear


%%%%%%%%%%%%%%%%%%%%%%%%%%% read in data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
times_std
times_powurv_gpu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
p1 = plot(n_std, t_svd ./ (n_std.^3), 'ks-',...
		n_std, t_cpqr ./ (n_std.^3), 'gx-',...
		n_rutv_gpu, t_rutv_gpu(1,:) ./ (n_rutv_gpu.^3), 'bo-',...
		n_rutv_gpu, t_rutv_gpu(2,:) ./ (n_rutv_gpu.^3), 'bv-.');
ax = gca;
set(ax,'fontsize',16);
set(p1,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Time [s] / n^3');
legend(	'MAGMA dgesdd (SVD)',...
		'MAGMA dgeqp3 (CPQR)',...
		'rand\_utv\_gpu, q=1',...
		'rand\_utv\_gpu, q=2');
legend('Location', 'NorthEast');
