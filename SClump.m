function [y, S, evs, A] = SClump(mp_matrix, c, true_cluster)
% mp_matrix: |mp| * n * n matrix, where |mp| is the number of meta-paths
% c: cluster number
% true_cluster: the true clustering results

% y: the final clustering result, cluster indicator vector
% S: learned similarity matrix
% evs: eigenvalues of learned graph Laplacian in the iterations
% A: learned symmetric similarity matrix

% Ref:
% Feiping Nie, Xiaoqian Wang, Michael I. Jordan, Heng Huang.
% The Constrained Laplacian Rank Algorithm for Graph-Based Clustering.
% The 30th Conference on Artificial Intelligence (\textbf{AAAI}), Phoenix, USA, 2016.

NITER = 20;
zr = 10e-11;

alpha = 0.5; % ||S||_F
beta = 10; % ||lambda||_2
gamma = 0.01; % trace

P = size(mp_matrix,1);% number of meta paths
n = size(mp_matrix,2);% number of objects
lambda = ones(P,1)./P;% initialization on the weights of meta paths

eps = 1e-10;

% A0 is the initial similarity matrix
A0 = zeros(n,n);
for p = 1:P
    A0 = A0 + lambda(p) * squeeze(mp_matrix(p,:,:));
end;

A0 = A0-diag(diag(A0));
A10 = (A0+A0')/2;
D10 = diag(sum(A10));
L0 = D10 - A10;

[F0, ~, evs]=eig1(L0, n, 0);
F = F0(:,1:c);
[pred] = postprocess(F,c,true_cluster);

for iter = 1:NITER
    dist = L2_distance_1(F',F');
    S = zeros(n);
    for i=1:n
        a0 = A0(i,:);
        idxa0 = 1:n;
        ai = a0(idxa0);
        di = dist(i,idxa0);
        ad = (ai-0.5*gamma*di)/(1+alpha); S(i,idxa0) = EProjSimplex_new(ad);
    end;
    A = S;
    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
    F_old = F;
    [F, ~, ev]=eig1(L, c, 0);
    [pred] = postprocess(F,c,true_cluster);
    evs(:,iter+1) = ev;

    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    
    lambda_old = lambda;
    
    if fn1 > zr
        gamma = 2*gamma;
        lambda =  optimizeLambda(mp_matrix, S, beta); % optimize lambda
    elseif fn2 < zr
        gamma = gamma/2;  F = F_old; lambda = lambda_old;
    else
        break;
    end;
    
    A0 = zeros(n,n);
    for p = 1:P
        A0 = A0 + lambda(p) * squeeze(mp_matrix(p,:,:));
    end;

end;

%[labv, tem, y] = unique(round(0.1*round(1000*F)),'rows');
[clusternum, y]=graphconncomp(sparse(A)); y = y';
nmi = calculateNMI(y,true_cluster);
purity = eval_acc_purity(true_cluster,y);
ri = eval_rand(true_cluster,y);

fprintf('Final NMI is %f\n',nmi);
fprintf('Final purity is %f\n',purity);
fprintf('Final rand is %f\n',ri);

if clusternum ~= c
    sprintf('Can not find the correct cluster number: %d', c)
end;





