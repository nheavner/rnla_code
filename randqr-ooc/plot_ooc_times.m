clear

%%%%%%%%%%%%%%%%%%%%%%%%%%% read in data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cpqr_ooc_times

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
p1 = plot(n_cpqr_ooc(1:length(t_cpqr_in)), t_cpqr_in ./ n_cpqr_ooc(1:length(t_cpqr_in)).^3,'ko-',...
		n_cpqr_ooc, t_cpqr_ssd ./ n_cpqr_ooc.^3, 'rs--',...
		n_cpqr_ooc(1:length(t_cpqr_hdd)), t_cpqr_hdd ./ n_cpqr_ooc(1:length(t_cpqr_hdd)).^3, 'bd-.');
ax = gca;
set(ax,'fontsize',16);
set(p1,'LineWidth',2,'MarkerSize',10);
xlabel('n');
ylabel('Time [s] / n^3');
legend(	'In Core',...
		'SSD',...
		'HDD');
legend('Location', 'East');
yticks([0e-10 0.5e-10 1e-10 1.5e-10 2e-10 2.5e-10 3e-10 3.5e-10])
set(ax,'YMinorTick','on')
ax.YAxis.MinorTickValues = [0.25e-10:0.5e-10:3e-10];
ax.YGrid =  'on'
ax.YMinorGrid = 'on'
