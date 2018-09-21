function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% size(Theta1) %25x401
% size(Theta2) %10x26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Part 1
%i think you have to implement forward prop and train it on each class in [0, 0, 0, 0, 0, 1, 0, 0, 0]



X = [ones(m,1) X];
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

%size(a3); % 5000 x 10

% used to be y./[1 2 3 4 5 6 7 8 9 10] == 1 but that doesnt handle any number of output nodes
y = y./(1:size(Theta2, 1)) == 1; % for the part3 in ex4.m you have to do 10 last. because it calculates the weight with 10 being last. otherwise im pretty sure the order doesnt matter at all.

J = y.*log(a3) + (1-y).*log(1-a3);
J = sum(sum(J));
J = -1*J/m;

% regularization
Theta1NoBias = Theta1(:,2:end);
Theta2NoBias = Theta2(:,2:end);

Theta1NoBias = sum(sum(Theta1NoBias.^2));
Theta2NoBias = sum(sum(Theta2NoBias.^2));

r = lambda/(2*m) * (Theta1NoBias + Theta2NoBias); % + Theta3NoBias + Theta4NoBias + ... Theta(L-1)NoBias
J = J + r;

%Part 2

for i=1:m,
    a1=X(i,:)'; % after transpose 400x1. our input nodes 
    d3=(a3(i,:)-y(i,:))'; % after transpose 10x1. errors for this training example
    l2z = [1 z2(i,:)]'; % after transpose 26x1. preactivation values for layer 2
    d2=(Theta2'*d3).*sigmoidGradient(l2z); % 26x1. this IS backprop
    d2=d2(2:end); % 25x1. get rid of bias 

    a2_i = a2(i,:); % 1x26. just grab it from the cost computation. 
    Theta2_grad += d3 * a2_i; % these two lines are just a vectorized implementation that automatically does update  
    Theta1_grad += d2 * a1';  % for all values of i (being training example #) and j (the jth node in that layer)
end

Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

%Part 3

Theta2_grad = Theta2_grad + (lambda / m) * ([zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]);
Theta1_grad = Theta1_grad + (lambda / m) * ([zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end















% ALL THAT FOLLOW ARE NOTES
% ######################################################################################################
% ######################################################################################################
% ######################################################################################################

%J = y'*log(a3) + (1-y')*log(1-a3); % whats the difference?????
%{
   ANSWER - IT WAS THE WRONG OPERATION.

    well for one in lrcostfunction you're doing 
    y=5000x1
    log(h)=5000x1
    so y'log(h) is a reasonable shorthand

    here we're training 10 output nodes at the same time so we have
    y=5000x10
    log(a3)=5000x10
    and we CANNOT use the same shorthand.

*you initially had this more or less figured out when you were just trying to match up with the picture of the equation
you got lost when it didnt work the first time and you reached for the lrcostfunction because you were confused.

   SIMPLE EXAMPLE.

   >> x=[1 2;3 4]
x =

   1   2
   3   4

>> y=[4 5;6 7]
y =

   4   5
   6   7

>> y'
ans =

   4   6
   5   7

>> y'*x
ans =

   22   32    <-- but i sum the products of every row vector in y times every col vector in x
   26   38

>> y.*x
ans =

    4   10   <----------- just a normal .*
   18   28

>>      
%}
