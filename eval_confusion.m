% Creates a confusion matrix for classification results
%
% Author: Frank Lin (frank@cs.cmu.edu)

function confusion=eval_confusion(truth,pred)

confusion=zeros(max(truth));

for i=1:length(truth)
   confusion(pred(i),truth(i))=confusion(pred(i),truth(i))+1; 
end

end