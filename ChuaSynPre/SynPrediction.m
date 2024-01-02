

clc;clear;close

load ChuaData
trainLen=length(inputdata(:,1));
inSize = 7; 
outSize = 6;
rng(144542361);


SRadius = Beyopt_result.SRadius; % the spectral radius
Win_a = Beyopt_result.Win_a; % the weighted matrix of the input data
a = Beyopt_result.a; % the leaking rate
reg = Beyopt_result.reg;  % the regularization coefficient
density = Beyopt_result.density; % the density of the adjacency matrix
resSize= Beyopt_result.resSize; % the size of reservoir nodes


Win = Win_a*(2.0*rand(resSize,inSize)-1.0);
W=sprand(resSize, resSize, density);
disp 'Computing spectral radius...';
opt.disp = 0;
RhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* (SRadius/RhoW); 

% run the reservoir with the training data to collect X
X = zeros(resSize,trainLen); 
Yt = youtdata'; % target output 
x = (2.0*rand(resSize,1)-1.0)*0.1; % reservoir state matrix

for t = 1:trainLen
    u = inputdata(t,:)';
    x = (1-a)*x + a*tanh( Win*u+W*x );
    x(1:2:end)=x(1:2:end).^2;
    X(:,t) = x;
end

n=100; % transient time
RNumber=2000;
X(:,2*RNumber+1:2*RNumber+n)=[];
Yt(:,2*RNumber+1:2*RNumber+n)=[];
X(:,RNumber+1:RNumber+n)=[];
Yt(:,RNumber+1:RNumber+n)=[];
X(:,1:n)=[];
Yt(:,1:n)=[];

% Coputing the output weight Wout
% X_T = X';
% Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
Wout=Yt*X'/(X*X'+reg*eye(resSize));



%%################## Synchronization predict with PARC ####################

WarmingTime=500;
m=6000; 
ErrorSpan=1000:m;
k=0;
SynError=NaN(2,41);
Yrc= zeros(outSize,m); 


for R=9:0.1:13  % coupling strength
    k=k+1;
    PA=R/13; % aware parameter
    
    for t = 1:WarmingTime 
        u=[inputdata(2000+t,1:6)'; PA];
        x=(1-a)*x + a*tanh( Win*u + W*x);
        x(1:2:end)=x(1:2:end).^2;
    end
    
    for i=1:m
        x=(1-a)*x + a*tanh( Win*u + W*x );
        x(1:2:end)=x(1:2:end).^2;
        Yrc(:,i) = Wout*x;
        u=[Yrc(:,i); PA];
    end
    
    err_sum=( Yrc(1,ErrorSpan)-Yrc(4,ErrorSpan) ).^2 ...
             +(Yrc(2,ErrorSpan)-Yrc(5,ErrorSpan)).^2 ...
             +( zmax*Yrc(3,ErrorSpan)-zmax*Yrc(6,ErrorSpan) ).^2;
    SynError(1,k)=R;
    SynError(2,k)=sqrt( mean(err_sum) );
    
end
clear R


% ============= Synchronization error VS coupling strength ================


figure(4)
plot(9:0.2:13,ExpSynErr(1,:)./10,'-ks',...
    'linewidth',1.5,...
    'markersize',8,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','none');
hold on
plot(SynError(1,:),SynError(2,:),'-or',...
    'LineWidth',1.5,...
    'markersize',6,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','none')
hold on
plot(9.4*ones(1,121),0:0.01:1.2,'--b','LineWidth',2)
hold on
plot(10.2*ones(1,121),0:0.01:1.2,'--b','LineWidth',2)
hold on
plot(11*ones(1,121),0:0.01:1.2,'--b','LineWidth',2)
hold off

% legend('Experiment','PARC')
xlabel('R6 (k\Omega)')
ylabel('RSME')
set(gca,'xlim',[8.8 13.2])
% set(gca,'ylim',[8.9 13.1])
set(gca,'FontSize',15,'FontName','Arial')
% set(gcf,'unit','centimeters','position',[8 2 18 14])
% set(gca,'Position',[.13 .18 .85 .80])



