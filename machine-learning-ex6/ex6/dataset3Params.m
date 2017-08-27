function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
Test = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ] ;
% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
predictionSet = zeros(8,8) ;
for c = 1:8
  C = Test(c) ;
  for s = 1:8
    sigma = Test(s) ;
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval) ;
    accuracy =  mean(double(predictions ~= yval)) ;
    predictionSet(c, s) = accuracy;
  end
end

m = min(min(predictionSet)) ;
f = (predictionSet == m) ;
[c1,s1] = find(f);
c = c1(1);
s = s1(1) ;
C = Test(c) 
sigma = Test(s) 

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







% =========================================================================

end
