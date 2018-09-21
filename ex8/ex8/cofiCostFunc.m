function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
% size(Theta) % 4x3
% size(X)     % 5x3
% size(Y)     % 5X4
% size(R)     % 5X4
%{

X (should be initialized to small random) =

   1.048686  -0.400232   1.194119 ---> movie1 feature vector
   0.780851  -0.385626   0.521198 ---> movie2 feature vector
   0.641509  -0.547854  -0.083796 ---> movie3 feature vector
   0.453618  -0.800218   0.680481 ---> movie4 feature vector
   0.937538   0.106090   0.361953 ---> movie5 feature vector

Theta (should be initialized to small random) =

   0.28544  -1.68427   0.26294 --> user1 params
   0.50501  -0.45465   0.31746 --> user2 params
  -0.43192  -0.47880   0.84671 --> user3 params
   0.72860  -0.27189   0.32684 --> user4 params

Y (user ratings for each movie) =

   u   u   u   u
   s   s   s   s
   e   e   e   e
   r   r   r   r
   1   2   3   4

   5   4   0   0    movie1
   3   0   0   0    movie2
   4   0   0   0    movie3
   3   0   0   0    movie4
   3   0   0   0    movie5

R =

    1   1   0   0
    1   0   0   0
    1   0   0   0
    1   0   0   0
    1   0   0   0

predictedRatings =

   u           u         u          u
   s           s         s          s
   e           e         e          e
   r           r         r          r
   1           2         3          4

   1.287418   1.090653   0.749762   1.263181   movie1
   1.009428   0.735125   0.288681   0.844126   movie2
   1.083812   0.546449  -0.085715   0.588972   movie3
   1.656187   0.808928   0.763394   0.770491   movie4
   0.184102   0.540142  -0.149265   0.772545   movie5

predictionErrors (how far off were our predictions? dont count movies the user hasnt rated) = 

  -3.71258  -2.90935   0.00000   0.00000
  -1.99057   0.00000   0.00000   0.00000
  -2.91619   0.00000  -0.00000   0.00000
  -1.34381   0.00000   0.00000   0.00000
  -2.81590   0.00000  -0.00000   0.00000

predictionErrorsSumOfSquares =

   13.78327    8.46430    0.00000    0.00000
    3.96238    0.00000    0.00000    0.00000
    8.50415    0.00000    0.00000    0.00000
    1.80583    0.00000    0.00000    0.00000
    7.92928    0.00000    0.00000    0.00000

J =  22.224604
%}

predictedRatings = X*Theta';
predictionErrors = (predictedRatings - Y).*R;
predictionErrorsSumOfSquares = predictionErrors.^2;
J = sum(sum(predictionErrorsSumOfSquares))/2;


%{

(X*Theta' - Y)' =

   m           m         m           m         m
   o           o         o           o         o
   v           v         v           v         v
   i           i         i           i         i
   e           e         e           e         e
   1           2         3           4         5

  -3.712582  -1.990572  -2.916188  -1.343813  -2.815898  user1
  -2.909347   0.735125   0.546449   0.808928   0.540142  user2
   0.749762   0.288681  -0.085715   0.763394  -0.149265  user3
   1.263181   0.844126   0.588972   0.770491   0.7725455 user4


X (X is still X) =

   1.048686  -0.400232   1.194119 ---> movie1 feature vector
   0.780851  -0.385626   0.521198 ---> movie2 feature vector
   0.641509  -0.547854  -0.083796 ---> movie3 feature vector
   0.453618  -0.800218   0.680481 ---> movie4 feature vector
   0.937538   0.106090   0.361953 ---> movie5 feature vector

X_grad     5x3    what? a lot is going on here.
Theta_grad 4x3    we're solving multiple optimization problems at once.
                  that's why the gradient isn't just nx1 like in linear 
                  regression from wk2
%}



X_grad = ((X*Theta' - Y).*R)*Theta;
Theta_grad = ((X*Theta' - Y).*R)'* X;

% REGULARIZATION
J += (lambda/2)*sum(sum(X.^2)); % literally just sum the whole table. before we just summed the parameter vector theta. now we sum the whole matrix.
J += (lambda/2)*sum(sum(Theta.^2)); % literally just sum the whole table. before we just summed the parameter vector theta. now we sum the whole matrix.
X_grad = X_grad + (lambda * X); % they're already laid out as a table. just add
Theta_grad = Theta_grad + (lambda * Theta); % they're already laid out as a table. just add
% =============================================================

grad = [X_grad(:); Theta_grad(:)]; % 27 x 1
end
