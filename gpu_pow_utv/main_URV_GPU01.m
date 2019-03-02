function main_URV_GPU01

%LOCAL_simple
LOCAL_plot

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_simple

rng(0)

n = 8000;

A  = randn(n);
B  = randn(n);
G  = randn(n);
gA = gpuArray(A);
gB = gpuArray(B);
gG = gpuArray(G);
q  = 2;

%%% Execute powerURV on CPU
tic
for i = 1:q
  G = A'*(A*G);
end
[V,~] = qr(G);
[U,R] = qr(A*V);
fprintf(1,'Time to execute on CPU = %10.3f\n',toc)

%%% Execute powerURV on GPU
tic
for i = 1:q
  gG = gA'*(gA*gG);
end
[gV,~] = qr(gG);
[gU,gR] = qr(gA*gV);
fprintf(1,'Time to execute on GPU = %10.3f\n',toc)

fprintf(1,'CPU: |A - U*R*trans(V)| = %12.5e\n',max(max(abs(A - U*R*V'))))
fprintf(1,'GPU: |A - U*R*trans(V)| = %12.5e\n',max(max(abs(gA - gU*gR*gV'))))

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_plot

rng(0)

nvec = 1000*(1:10);
tturv0 = zeros(size(nvec));
tturv1 = zeros(size(nvec));
tturv2 = zeros(size(nvec));
ttsvd  = zeros(size(nvec));
tturv0gpu = zeros(size(nvec));
tturv1gpu = zeros(size(nvec));
tturv2gpu = zeros(size(nvec));
ttsvdgpu   = zeros(size(nvec));
%ee1  = zeros(size(nvec));
%ee2  = zeros(size(nvec));

for icount = 1:length(nvec)
  n  = nvec(icount);
  A  = randn(n);
  G  = randn(n);
  gA = gpuArray(A);
  gG = gpuArray(G);
  tic
  [U,D,V] = svd(A);
  ttsvd(icount) = toc;
  tic
  [U,R,V] = LOCAL_PURV(A,G,0);
  tturv0(icount) = toc;
  tic
  [U,R,V] = LOCAL_PURV(A,G,1);
  tturv1(icount) = toc;
  tic
  [U,R,V] = LOCAL_PURV(A,G,2);
  tturv2(icount) = toc;
  tic
  [gU,gD,gV] = svd(gA);
  ttsvdgpu(icount) = toc;
  tic
  [gU,gR,gV]  = LOCAL_PURV(gA,gG,0);
  tturv0gpu(icount) = toc;
  tic
  [gU,gR,gV]  = LOCAL_PURV(gA,gG,1);
  tturv1gpu(icount) = toc;
  tic
  [gU,gR,gV]  = LOCAL_PURV(gA,gG,2);
  tturv2gpu(icount) = toc;
  fprintf(1,'finished n = %d\n',n)
end

figure(1)
subplot(1,2,1)
plot(nvec,ttsvd./(nvec.^3),'b.-',...
     nvec,tturv0./(nvec.^3),'b',...
     nvec,tturv1./(nvec.^3),'b:',...
     nvec,tturv2./(nvec.^3),'b--',...
     nvec,ttsvdgpu./(nvec.^3),'r.-',...
     nvec,tturv0gpu./(nvec.^3),'r',...
     nvec,tturv1gpu./(nvec.^3),'r:',...
     nvec,tturv2gpu./(nvec.^3),'r--',...
     min(nvec),0)
xlabel('n')
ylabel('t/n^3')
legend('ttsvd','tturv0','tturv1','tturv2','ttsvdgpu','tturv0gpu','tturv1gpu','tturv2gpu')
axis tight
    
subplot(1,2,2)
loglog(nvec,ttsvd,'b.-',...
       nvec,tturv0,'b',...
       nvec,tturv1,'b:',...
       nvec,tturv2,'b--',...
       nvec,ttsvdgpu,'r.-',...
       nvec,tturv0gpu,'r',...
       nvec,tturv1gpu,'r:',...
       nvec,tturv2gpu,'r--')
xlabel('n')
ylabel('t')
legend('ttsvd','tturv0','tturv1','tturv2','ttsvdgpu','tturv0gpu','tturv1gpu','tturv2gpu')
axis tight
    
%subplot(1,2,2)
%plot(nvec,ee1,nvec,ee2)

fprintf(1,'    n      ttsvd     tturv0     tturv1     tturv2   ttsvdgpu  tturv0gpu  tturv1gpu  tturv2gpu\n')
for i = 1:length(nvec)
  fprintf(1,'%5d  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f\n',...
          nvec(i),ttsvd(i),tturv0(i),tturv1(i),tturv2(i),ttsvdgpu(i),tturv0gpu(i),tturv1gpu(i),tturv2gpu(i))
end

keyboard

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U,R,V] = LOCAL_PURV(A,G,q)

for i = 1:q
  G = A'*(A*G);
end
[V,~] = qr(G);
[U,R] = qr(A*V);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
