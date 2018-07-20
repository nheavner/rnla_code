clear


%%%%%%%%%%%%%%%%%%%%%%%%%%% read in data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
profile_rutv_cpu_dat
profile_rutv_gpu_dat

total_time_cpu = sum(profile_b128_q0_cpu,2);
total_time_gpu = sum(profile_b128_q0_gpu,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,1,1)
p1 = plot(n, profile_b128_q0_cpu(:,1)' ./ total_time_cpu' * 100, 'k-',...
		  n, profile_b128_q0_cpu(:,2)' ./ total_time_cpu' * 100, 'b-',...
		  n, profile_b128_q0_cpu(:,3)' ./ total_time_cpu' * 100, 'g-',...
		  n, profile_b128_q0_cpu(:,4)' ./ total_time_cpu' * 100, 'r-');
ax = gca;
set(ax,'fontsize',16);
set(p1,'LineWidth',2,'MarkerSize',10);
ylim([0 100])
title('CPU');
xlabel('n');
ylabel('Relative time spent [%]');
legend(	'build\_y',...
		'qr\_1',...
		'qr\_2',...
		'svd');
legend('Location', 'SouthWest');

subplot(2,1,2)
p1 = plot(n, profile_b128_q0_gpu(:,1)' ./ total_time_gpu' * 100, 'k-',...
		  n, profile_b128_q0_gpu(:,2)' ./ total_time_gpu' * 100, 'b-',...
		  n, profile_b128_q0_gpu(:,3)' ./ total_time_gpu' * 100, 'g-',...
		  n, profile_b128_q0_gpu(:,4)' ./ total_time_gpu' * 100, 'r-');
ax = gca;
set(ax,'fontsize',16);
set(p1,'LineWidth',2,'MarkerSize',10);
ylim([0 100])
title('GPU');
xlabel('n');
ylabel('Relative time spent [%]');
legend(	'build\_y',...
		'qr\_1',...
		'qr\_2',...
		'svd');
legend('Location', 'SouthWest');

figure();
subplot(2,1,1)
p1 = semilogy(n, profile_b128_q0_cpu(:,1)' ./ n.^3, 'k-',...
		  n, profile_b128_q0_cpu(:,2)'./ n.^3, 'b-',...
		  n, profile_b128_q0_cpu(:,3)' ./ n.^3, 'g-',...
		  n, profile_b128_q0_cpu(:,4)' ./ n.^3, 'r-');
ax = gca;
set(ax,'fontsize',16);
set(p1,'LineWidth',2,'MarkerSize',10);
title('CPU');
xlabel('n');
ylabel('Time [sec] / n^3');
legend(	'build\_y',...
		'qr\_1',...
		'qr\_2',...
		'svd');
legend('Location', 'SouthWest');

subplot(2,1,2)
p1 = semilogy(n, profile_b128_q0_gpu(:,1)' ./ n.^3, 'k-',...
		  n, profile_b128_q0_gpu(:,2)' ./ n.^3, 'b-',...
		  n, profile_b128_q0_gpu(:,3)' ./ n.^3, 'g-',...
		  n, profile_b128_q0_gpu(:,4)' ./ n.^3, 'r-');
ax = gca;
set(ax,'fontsize',16);
set(p1,'LineWidth',2,'MarkerSize',10);
title('GPU');
xlabel('n');
ylabel('Time [sec] / n^3');
legend(	'build\_y',...
		'qr\_1',...
		'qr\_2',...
		'svd');
legend('Location', 'SouthWest');
