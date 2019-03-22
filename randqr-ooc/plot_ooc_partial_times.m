clear

%%%%%%%%%%%%%%%%%%%%%%%%%%% read in data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cpqr_ooc_partial_times


%%%%%%%%%% generate line vals for predicted ssd performance  %%%%%%%%%%%
% NOTE: This function will change depending on k,b
k = 1000;
b = 250;
mat_size = n_cpqr_ooc.^2*8/(1e+09);
t_cpqr_ssd_predicted_comm =  ( 2*mat_size + b./n_cpqr_ooc.*mat_size ) ./ 4.4 + ... % read  
				mat_size ./ 1.4; % write
t_cpqr_ssd_predicted_flop = (t_cpqr_in(end) ./ n_cpqr_ooc(length(t_cpqr_in))^2) .* n_cpqr_ooc.^2;

t_cpqr_ssd_predicted = t_cpqr_ssd_predicted_comm + t_cpqr_ssd_predicted_flop;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
p1 = plot(n_cpqr_ooc(1:length(t_cpqr_in)), t_cpqr_in ./ n_cpqr_ooc(1:length(t_cpqr_in)).^2,'ko-',...
		n_cpqr_ooc, t_cpqr_ssd ./ n_cpqr_ooc.^2, 'rs--',...
		n_cpqr_ooc(1:length(t_cpqr_hdd)), t_cpqr_hdd ./ n_cpqr_ooc(1:length(t_cpqr_hdd)).^2, 'rs-.',...
		n_cpqr_ooc_pp, t_cpqr_ssd_pp ./ n_cpqr_ooc_pp.^2, 'bd--',...
		n_cpqr_ooc_pp(1:length(t_cpqr_hdd_pp)), t_cpqr_hdd_pp ./ n_cpqr_ooc_pp(1:length(t_cpqr_hdd_pp)).^2, 'bd-.',...
		n_cpqr_ooc_left, t_cpqr_ssd_left ./ n_cpqr_ooc_left.^2, 'mv--',...
		n_cpqr_ooc_left(1:length(t_cpqr_hdd_left)), t_cpqr_hdd_left ./ n_cpqr_ooc_left(1:length(t_cpqr_hdd_left)).^2, 'mv-.');
ax = gca;
set(ax,'fontsize',16);
set(p1,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Time [s] / n^2');
title('Partial Factorization Times, k=1000, b=250')
legend(	'In Core',...
		'In place, SSD',...
		'In place, HDD',...
		'Physical pivoting, SSD',...
		'Physical pivoting, HDD',...
		'Left-looking, SSD',...
		'Left-looking, HDD' );
legend('Location', 'northwest');
%yticks([0e-10 0.5e-10 1e-10 1.5e-10 2e-10 2.5e-10 3e-10 3.5e-10])
set(ax,'YMinorTick','on')
ax.YAxis.MinorTickValues = [0.25e-10:0.5e-10:3e-10];
ax.YGrid =  'on'
ax.YMinorGrid = 'on'
