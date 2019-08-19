% Accuracy evaluation: returns prediction purity given predicted and true
% labels
%
% Author: Frank Lin (frank@cs.cmu.edu)

function acc=eval_acc_purity(truth,pred)

confusion=eval_confusion(truth,pred);

acc=sum(max(confusion,[],2))/length(truth);

end