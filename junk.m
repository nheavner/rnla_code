f = fopen('q0_1core.txt','w');

fprintf(f,'\t no update \t update \n');

for i=1:length(noupdate_time)
    fprintf(f,'n = %i:\t %.2e \t %.2e \n',n(i),noupdate_time(i),update_time(i)); 
end

fclose(f);
