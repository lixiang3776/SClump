function [lambda] =  optimizeLambda(mp_matrix, S, beta)

P = size(mp_matrix,1);% the number of meta paths

X = sym('x',[1 P]);
assume(X, 'real');

Q = 0;

for i = 1:P
    Sp = squeeze(mp_matrix(i,:,:));
    Q = Q - 2 * X(i) * trace(S'*Sp);
end;

for i = 1:P
    Sp1 = squeeze(mp_matrix(i,:,:));
    Q = Q + X(i) * X(i) * trace(Sp1'*Sp1);% quadratic
    for j = (i+1):P
        Sp2 = squeeze(mp_matrix(i,:,:));
        Q = Q + 2 * X(i) * X(j) * trace(Sp1'*Sp2);
    end;
end;

for i = 1:P
    Q = Q + sym(sprintf('%d',beta)) * X(i)^2;
end;

global g;
g = matlabFunction(Q);

lambda = [];
exitflag = 0;

%initialization
Aeq = ones(1,P);
beq = 1;
% Aeq = [];
% beq = [];
lb = zeros(1,P);
ub = ones(1,P);
nonlcon = [];
options = optimoptions('fmincon','MaxFunctionEvaluations',4000,'ObjectiveLimit',-1e40);

while 1
    x0 = rand(1,P);
    
    [x, fval, exitflag] = fmincon(@fun,x0,[],[],Aeq,beq,lb,ub,nonlcon,options);
    
%     if abs(min(val_ref) - fval)/abs(fval) > 1e-5
%         sprintf('%f',min(val_ref))
%         sprintf('%f',fval)
%         continue;
%     end
    
    if exitflag == 1 || exitflag == 2
        lambda = x';
        disp(lambda);
        disp(fval);
        disp(exitflag);
        break;
    else
        disp(exitflag);
    end;
end;

end

function f = fun(y)
x = {};
for i=1:length(y)
    x = [x, y(i)];
end;

global g;

f = g(x{:});
end




