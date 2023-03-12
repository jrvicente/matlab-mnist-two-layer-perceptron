function y = dLeakyReLU(x, alpha)
% simpleLogisticSigmoid Logistic sigmoid activation function
%
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector where the logistic sigmoid was applied element by
% element.
%

y = zeros(size(x));
y(x>=0) = 1;
y(x<0) = alpha;

end