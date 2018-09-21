function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    predictions = (pval < epsilon);

%{  
        truePositives  = sum((predictions == 1) & (yval == 1));
        
    a note about these. you cant use

        truePositives  = sum((predictions == 1) == (yval == 1));

    because of basic programming, not anything special about octave
    & works and == doesnt because == will evaluate to true if 
    predictions != 1 and yval != 1 whereas & only evaluates to
    true if predictions and yval == 1.
%}
    truePositives  = sum((predictions == 1) & (yval == 1));
    falsePositives = sum((predictions == 1) & (yval == 0));
    falseNegatives = sum((predictions == 0) & (yval == 1));
    
    numberOfCorrectPredictions = truePositives;
    numberOfPredictedAnomalies = truePositives + falsePositives;
    numberOfAnomalies = truePositives + falseNegatives; % e.g. sum(yval == 1)

    if(numberOfPredictedAnomalies == 0)
        precision = 0;
    else
        precision = numberOfCorrectPredictions/numberOfPredictedAnomalies; % how accurate we were
    endif

    if(numberOfAnomalies == 0)
        recall = 0
    else
        recall = numberOfCorrectPredictions/numberOfAnomalies; % of all the anomalies. how many did we predict
    endif

    if((precision + recall) == 0)
        F1 = 0;
    else
        F1 = (2*precision*recall)/(precision + recall);
    endif

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
