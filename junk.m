f = fopen('q0_4core_matlab.txt','w');

fprintf(f,'\t no update \t update \t nuclear norm\n');

for i=1:length(noupdate_time)
    fprintf(f,'n = %i:\t %.2e \t %.2e \t %.2e \n',n(i),noupdate_time(i),update_time(i),nuc_norm_time(i)); 
end

fprintf(f,'\n \n \t Acceleration (no update time / nuclear norm time) \n');

for i=1:length(noupdate_time)
    fprintf(f,'n = %i:\t %.2e \n',n(i),accel(i)); 
end

fclose(f);
