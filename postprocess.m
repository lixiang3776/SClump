function [pred,nmi,purity,ri] = postprocess(f,k,trueLabel)

%normalized by row
for i=1:size(f,1);
    f(i,:)=f(i,:)/(norm(f(i,:))+1e-10);
end

pred=kmeans_freq(f,k,10,'m');

% pred = kmeans(f,k);

% count = zeros(k,1);
% for i =1:size(pred,1)
%     count(pred(i)) = count(pred(i))+1;
% end
% for i = 1:k
%     disp(count(i));
% end

% n = size(f,1);
% cluster = zeros(n,k);
% for i = 1:n
%     cluster(i,pred(i)) = 1;
% end;

nmi = calculateNMI(pred,trueLabel);
purity=eval_acc_purity(trueLabel,pred);
ri=eval_rand(trueLabel,pred);

fprintf('NMI is %f\n',nmi);
fprintf('purity is %f\n',purity);
fprintf('rand is %f\n',ri);
