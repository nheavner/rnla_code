profile_bl_size_data_25k

figure()

subplot(2,1,1)
plot(b,t_tot,'kd-',b,t_comm,'ro-',b,t_flop,'bs-')
xlabel('block size')
ylabel('time [s]')
title('Profiling data for ooc cpqr, n=25000')
legend('total time','comm time','flop time','Location','East')

subplot(2,1,2)
plot(b,perc_comm,'ro-',b,perc_flop,'bs-')
xlabel('block size')
ylabel('[%]')
legend('percentage comm','percentage flop','Location','East')
