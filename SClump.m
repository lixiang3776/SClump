% min_{S>=0, S*1=1, F'*F=I}  ||S - A||^2 + r*||S||^2 + 2*lambda*trace(F'*L*F)
% or
% min_{S>=0, S*1=1, F'*F=I}  ||S - A||_1 + r*||S||^2 + 2*lambda*trace(F'*L*F)
function [y, S, evs, A, lambdas, gammas] = SClump(mp_matrix, c, true_cluster)
% A0: the given affinity matrix
% c: cluster number
% isrobust: solving the second (L1 based) problem if isrobust=1
% islocal: only update the similarities of neighbors if islocal=1
% y: the final clustering result, cluster indicator vector
% S: learned symmetric similarity matrix
% evs: eigenvalues of learned graph Laplacian in the iterations
% cs: suggested cluster numbers, effective only when the cluster structure is clear

% Ref:
% Feiping Nie, Xiaoqian Wang, Michael I. Jordan, Heng Huang.
% The Constrained Laplacian Rank Algorithm for Graph-Based Clustering.
% The 30th Conference on Artificial Intelligence (\textbf{AAAI}), Phoenix, USA, 2016.

% series: 1,10,0.01
% restaurant: 0.1,10,0.01
% dblp: 0.1,10,1
% shop: 0.1,10,1

NITER = 20;
zr = 10e-11;

alpha = 0.5; % ||S||_F
beta = 10; % ||lambda||_2
gamma = 0.01; % trace

P = size(mp_matrix,1);% number of meta paths
n = size(mp_matrix,2);% number of objects
lambda = ones(P,1)./P;% initialization on the weights of meta paths

lambdas = lambda;
gammas = gamma;

eps = 1e-10;

A0 = zeros(n,n);
for p = 1:P
    A0 = A0 + lambda(p) * squeeze(mp_matrix(p,:,:));
end;

A0 = A0-diag(diag(A0));
% B0 = A0 ./ repmat(sum(A0,2), 1, n);
A10 = (A0+A0')/2;
D10 = diag(sum(A10));
L0 = D10 - A10;

% automatically determine the cluster number
[F0, ~, evs]=eig1(L0, n, 0);
F = F0(:,1:c);
[pred] = postprocess(F,c,true_cluster);

for iter = 1:NITER
    dist = L2_distance_1(F',F');
    S = zeros(n);
    for i=1:n
        a0 = A0(i,:);
        idxa0 = 1:n;
%         if islocal == 1
%             idxa0 = find(a0>0);
%         else
%             idxa0 = 1:num;
%         end;
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
        lambda =  optimizeLambda(mp_matrix, S, beta);
    elseif fn2 < zr
        gamma = gamma/2;  F = F_old; lambda = lambda_old;
    else
        break;
    end;
    
%     lambdas = [lambdas,lambda];
%     gammas = [gammas;gamma];
    
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





