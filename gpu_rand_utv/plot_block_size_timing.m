clear


%%%%%%%%%%%%%%%%%%%%%%%%%%% read in data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
block_size_timing_gpu_dat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
p1 = plot(b, profile_q0_n10k_gpu(:,1)', 'k-',...
		  b, profile_q0_n10k_gpu(:,2)', 'b-',...
		  b, profile_q0_n10k_gpu(:,3)', 'g-',...
		  b, profile_q0_n10k_gpu(:,4)', 'r-',...
		  b, profile_q0_n10k_gpu(:,5)', 'c-',...
		  b, profile_q0_n10k_gpu(:,6)', 'm-',...
		  b, profile_q0_n10k_gpu(:,7)', 'k--*');
ax = gca;
set(ax,'fontsize',16);
set(p1,'LineWidth',2,'MarkerSize',10);
title('Timing data for rand\_utv\_gpu, p=0, n=10000, q=0');
xlabel('block size');
ylabel('Time [s]');
legend(	'build\_y',...
		'qr\_1',...
		'qr\_1\_updt\_a',...
		'qr\_2',...
		'qr\_2\_updt\_a',...
		'svd',...
		'total' );
legend('Location', 'SouthWest');
