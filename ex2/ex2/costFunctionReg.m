function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% hype, or just call costFunction.m
h = X*theta;
h = sigmoid(h);
J = -y'*log(h) - (1-y')*log(1-h);
J = J/m;

% the regularization part
thetaR = theta(2:size(theta));
L = (lambda/(2*m))*thetaR'*thetaR;
J = J + L;

% theta 0 uses different function. too lazy to do this smart.
% i think in the video he said it doesn't really matter if you treat t0 different
% in which case i think you can just do ~~~~~ (X'*(h - y)+lambda*thetaR)/m;

t0 = (     (X'*(h - y))/m     )(1); % get the first element, 1
tg0 = ((     (X'*(h - y))(2:end)     )+lambda*thetaR)/m; % operate on 2-28

grad(1) = t0;
grad(2:end) = tg0;


% =============================================================

end
