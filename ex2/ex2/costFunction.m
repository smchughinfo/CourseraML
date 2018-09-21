function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h = X*theta;
h = sigmoid(h);

% HOW/WHY TO MULTIPLY Y AS A VECTOR:
%
% h(x)= θ + θX2 + θX3
% X = 1 2 3
%     9 0 4
% θ = 4
%     5
%     6
% y = 2
%     3
% 1(4) + 2(5) + 3(6) = 32
% 9(4) + 0(5) + 4(6) = 60
%
% J(θ) = 1/2m ∑[-yⁱln(hθ(xⁱ)) - (1-yⁱ)ln(1-hθ(xⁱ))]
%
%
%     y      h (the variable h, at this point.)
%     |      |
%     ^   ________
%    | | |        |
%    -yⁱ  ln(hθ(xⁱ))
% 
%  COST IS SCALAR, NOT VECTOR
%
% if you use .* to multiply it looks like this
%
%  -2    32  = -64
%  -3    60    -180
%
% but if you do it as a vector you still multiply the ith h(x) by the ith y
%
%  -2 -3   32 = -244
%          60
%
% *I suppose you could sum them manually but this is vectorized.
%
% *You get away with turning both the left and right side of this next equation
% into a scalar right in the middle of the equation because one side always 
% becomes 0 so it doesn't matter

J = -y'*log(h) - (1-y')*log(1-h);
J = J/m;


% "gradient" here is just the partial derivative of each member of θ
% it's used by fminunc or your optimization algorithm of choice
% we are returning the cost, think point on 3D graph
% and each member of θ's contribution to the error
% whatever algorithm we use is responsible for adjusting θ so the cost decreases

grad = (X'*(h - y))/m; % dividing (X'*(h - y)) by m is cool because you can divide a vector by a scalar

end
