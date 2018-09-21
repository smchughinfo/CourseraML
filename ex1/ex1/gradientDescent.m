function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h = X*theta; % just like cost func
    J = h - y; % just like cost func

#{  
        this is where tricky part starts
        remember it's that above *
    J the ith x_j
        if you transpose the original X and multiply it by what J currently is
        you get

        X
        1 3
        1 5            <-- ith
        1 9
        1 5

        theta
        0
        0

        x'
        1 1 1 1
        3 5 9 5

        J (in its present form looks something like...)
        8
        3
        2
        5

        X'J looks like:
        1 1 1 1       8
        3 5 9 5       3
                      2
                      5



      
        J_0 --> 8*1 + 3*1 + 2*1 + 5*1 = 21
        J_1 --> 8*3 + 3*5 + 2*9 + 5*5 = 82

                       ith

                        |
                        ∨
                       ___
                       |1|
                       |5|
                       ‾‾‾

        multiply each parameter by the corresponding x values that were used to calculate J
        youre basically just going back through the summation loop again to multiply the result from
        (h() -y) by x^i_j

        what all this does is adjust each parameter in theta by its contribution to the error
#}

#   h = X*theta; % just like cost func
#   J = h - y; % just like cost func

    J = X'*J;
    J = J/m;
    J = J*alpha;
    theta = theta - J;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    end
end
