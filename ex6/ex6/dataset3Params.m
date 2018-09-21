function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

lowestError = 100;
lowestC = 100;
lowestSigma = 100;
for curC=1:4
    for curSigma=1:4
        oneC = .001*10^curC;
        threeC = .003*10^curC;
        oneSigma = .001*10^curSigma;
        threeSigma = .003*10^curSigma;

        oneOneModel = svmTrain(X, y, oneC, @(x1, x2) gaussianKernel(x1, x2, oneSigma));
        oneThreeModel = svmTrain(X, y, oneC, @(x1, x2) gaussianKernel(x1, x2, threeSigma));
        threeThreeModel = svmTrain(X, y, threeC, @(x1, x2) gaussianKernel(x1, x2, threeSigma));
        threeOneModel = svmTrain(X, y, threeC, @(x1, x2) gaussianKernel(x1, x2, oneSigma));

        oneOnePrediction = svmPredict(oneOneModel, Xval);
        oneThreePrediction = svmPredict(oneThreeModel, Xval);
        threeThreePrediction = svmPredict(threeThreeModel, Xval);
        threeOnePrediction = svmPredict(threeOneModel, Xval);

        oneOneError = mean(double(oneOnePrediction ~= yval));
        oneThreeError = mean(double(oneThreePrediction ~= yval));
        threeThreeError = mean(double(threeThreePrediction ~= yval));
        threeOneError = mean(double(threeOnePrediction ~= yval));

        oneOneError
        oneThreeError
        threeThreeError
        threeOneError

        if(oneOneError < lowestError) 
            lowestError = oneOneError;
            lowestC = oneC;
            lowestSigma = oneSigma;
        endif
        if(oneThreeError < lowestError) 
            lowestError = oneThreeError;
            lowestC = oneC;
            lowestSigma = threeSigma;
        endif
        if(threeThreeError < lowestError) 
            lowestError = threeThreeError;
            lowestC = threeC;
            lowestSigma = threeSigma;
        endif
        if(threeOneError < lowestError) 
            lowestError = threeOneError;
            lowestC = threeC;
            lowestSigma = oneSigma;
        endif
    end
end

C = lowestC;
sigma = lowestSigma;
% =========================================================================

end
