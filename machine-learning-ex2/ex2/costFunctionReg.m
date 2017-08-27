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
z = X * theta ;
g = sigmoid(z) ;
toS = (-y)' * log(g) - ( (1-y)' * log(1 - g) ) ;
m = length(X) ;
theta(1, :) = [0] ; 
%Jtheta = theta( [2:m], :) ;
%Jtheta = [ones(1, 1), Jtheta] ;
J = ((1/m) * toS )+ ((lambda/(2 * m)) * sum(theta.^2)); 
test = sum(theta.^2)

gToS =  X' * (g-y);
grad = (1/m) * gToS + ((lambda/m) * theta);

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% =============================================================

end
