function g = sigmoid(z) 
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% scalar / hello world
% 1/(1+e^(-z))
% vector seems to be the most appropriate
% ...doing it as a vector works for scalar anyways

g = e.^(-z);
g = g + 1;
g = 1 ./ g;

% =============================================================

end
