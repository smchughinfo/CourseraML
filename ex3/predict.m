function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%{
X 5000x401     Theta1 25x401
3 4 5          2 3 4
2 4 5          3 4 4
2 3 4
1 4 5         
              
X' 401x5000    Theta1' 401x25
3 2 3 1        2 3
4 4 3 4        3 4  
5 5 4 5        4 4

This is just matrix multiplication. X is 5000 rows with 401 features.
We can line up and multiply Theta1 if we transpose it.
Theta1'
Now, Theta1 has its 401 features in each column.
Do that for all 25 columns and you have 25 thetas for that training example in the next layer
Do that for all 5000 training examples and you get a 5000x25 matrix 

...same rules for subsequent layers.

%}

X = [ones(m,1) X]; % add ones for bias
a2 = X*Theta1';
a2 = sigmoid(a2);
a2 = [ones(size(a2, 1),1) a2]; % add ones for bias
a3 = a2*Theta2';
a3 = sigmoid(a3);

[p_max, i_max]=max(a3, [], 2);
p = i_max;





% =========================================================================


end
