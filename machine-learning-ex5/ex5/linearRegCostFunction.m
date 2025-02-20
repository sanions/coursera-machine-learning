function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = X * theta;
JWithoutRegular = (1/ (2*m)) * sum(sum((h-y).^2)) ;
thetaRegular = theta(2:end) ;
RegularTerm = sum(thetaRegular.^2)  * lambda/(2*m) ;
J = JWithoutRegular + RegularTerm ;


gradForThetaZero = 1/m * sum(sum(h-y)) ;
grad(1) = gradForThetaZero ;

XForGrad = X(:,(2:end)) ;
grad_2 = 1/m * (h-y)'* XForGrad ;
grad_2RegularTerm = (lambda/m) * thetaRegular ;
grad(2:end) = grad_2 + grad_2RegularTerm;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% =========================================================================

grad = grad(:);

end
