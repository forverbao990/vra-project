function DisplayConfusionMatrix(confMat, digits)
% Display the confusion matrix in a formatted table.
% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '      |']);
    fprintf('%-9.2f', confMat(idx,:) * 100);
    fprintf('\n')
end