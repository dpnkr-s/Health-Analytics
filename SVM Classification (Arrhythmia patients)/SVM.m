%% Support Vector Machine


%% Data Preparation

load('arrhythmia.mat','arrhythmia');
s = sum(arrhythmia);
empty_col=find(s==0);
count = 0;

arrhythmia_old = arrhythmia;
arrhythmia(:,empty_col) = [];
arrhythmiaAll=arrhythmia;
iii=find(arrhythmia(:,end)>2);
arrhythmia(iii,end)=2;

%% Using 'linear' kernel

y1 = arrhythmia(:,1:end-1);
c = arrhythmia(:,end);
[N,F] = size(y1);
ymean = mean(y1);
yvar = var(y1);
o = ones(N,1);
y = (y1-o*ymean)./sqrt(o*yvar);

bc = 0.01;
for i=1:1000
    Mdl=fitcsvm(y,c,'BoxConstraint',bc,'KernelFunction','linear');
    classhat=sign(y*Mdl.Beta+Mdl.Bias);

    CVMdl = crossval(Mdl);
    classLoss = kfoldLoss(CVMdl);
    
    cl2(i) = {classLoss};
    bc=bc+0.01;
end

cl2=cell2mat(cl2);
[m,id] = min(cl2);
idx = 0.01:0.01:10;
plot(idx,cl2)

% Firstly, searching for minimun at box constraint value between 0.1 and 10
% ... at interval of 0.1,
% result: minimum is at 0.1
% Then, searching for minimum between 0.01 and 0.2, at interval of 0.01,
% result: minimum is at 0.04

%% Using 'gaussian' kernel
% bc = 5; % gives the best result so far
% SVMg=fitcsvm(y,c,'BoxConstraint',5,'KernelFunction','gaussian','KernelScale','auto');
% CVSVMg = crossval(SVMg);
% gclassLoss = kfoldLoss(CVSVMg);
% 
% clear cl2;

% It is observed so far, linear kernel performs better than gaussian
% kernel, as classloss is lesser in linear kernel 