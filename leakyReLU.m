function y = leakyReLU(x, alpha)
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
y(x>=0) = x(x>=0);
y(x<0) = alpha.*x(x<0);

end