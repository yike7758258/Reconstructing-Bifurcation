

clc;clear;

load chua.mat Y
load opt.mat hyperpara_set

eig_rho = hyperpara_set(1); % the spectral radius of the adjacency matrix
W_in_a = hyperpara_set(2); % weighted matrix of the input data
reg = hyperpara_set(3); % the regression coefficient
density = hyperpara_set(4); % the density of the adjacency matrix 
W_in_b = hyperpara_set(5); % weighted matrix of the aware parameters
a = hyperpara_set(6); % the leaking rate


rng(172166179)
Ydata=Y;
inSize = 4; 
outSize = 3;
indata=Ydata(:,1:inSize);
outdata=Ydata(:,1:outSize);
trainLen =12000;
testLen = 25000;
resSize =600; % size of the reservoir nodes;  

Win = 2.0*rand(resSize, inSize)-1;
Win(1:resSize, 1:outSize)=Win(1:resSize, 1:outSize)*W_in_a;
Win(1:resSize, inSize)=(2.0*rand(resSize,1)-1.0)*W_in_b;

WW = zeros(resSize,resSize); % weighted adjacency matrix of reservoir nodes
for i=1:resSize
    for j=i:resSize
            if (rand()<density)
                 WW(i,j)=(2.0*rand()-1.0);
                 WW(j,i)=WW(i,j);
            end
    end
end
rhoW = abs(eigs(WW,1));
W = WW .* (eig_rho /rhoW); 

X = zeros(resSize,trainLen);
% set the corresponding target matrix directly
Yt = outdata(2:trainLen+1,:)';
% run the reservoir with the data and collect X
x = (2.0*rand(resSize,1)-1.0);

for t = 1:trainLen
    u = indata(t,:)';
    x = (1-a)*x + a*tanh( Win*u + W*x );
    X(:,t) = x;
    X(1:2:end,t) = X(1:2:end,t).^2; 
end

X(:,8000:8100)=[];
Yt(:,8000:8100)=[];
X(:,4000:4100)=[];
Yt(:,4000:4100)=[];
X(:,1:100)=[];
Yt(:,1:100)=[];

% train the output weights Wout
X_T = X';
Wout = Yt*X_T / (X*X_T + reg*eye(resSize));


% Reconstructing bifurcation diagrams with the  parameter-aware reservoir computing 
discard=100;
Y1= zeros(outSize,testLen);
startL=0000;
x1=0;x2=0;
bifurcation=[];

for eps=1730:0.1:1770   % Bifurcation parameters
    
    for t = 1:discard 
        u = Ydata(t+startL,:)';
        x = (1-a)*x + a*tanh( Win*u + W*x ); 
    end
    u = Ydata(t+startL+1,:)';
    
    
    for t = 1:10000 
        x = (1-a)*x + a*tanh( Win*u + W*x );
        xx = x;
        xx(1:2:end) = xx(1:2:end).^2; 
        y = Wout*xx;
        u(1:outSize) = y;
        u(inSize)=eps;
        x2=x1;
        x1=y(1);
    end
    
    
    for t = 1:testLen 
        x = (1-a)*x + a*tanh( Win*u + W*x );
        xx = x;
        xx(1:2:end) = xx(1:2:end).^2; 
        y = Wout*xx;
        u(1:outSize) = y;
        u(inSize)=eps;
        if (x1<x2 && x1<y(1))
            bifurcation=[bifurcation;eps,x1];
        end
        x2=x1;
        x1=y(1);
    end

end


%%Bifurcation diagrams reconstructed with PARC
time=0:testLen-1;
figure(1);
plot(bifurcation(:,1),bifurcation(:,2),'k.','MarkerSize',0.2);
xlabel('\it{T}','FontName','Times New Roman','FontSize',24);
ylabel('\it{x}','FontName','Times New Roman','FontSize',24);



