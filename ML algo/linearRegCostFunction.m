function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% Linear regression cost function with regularisation and gradient descent (vectorised implementation) 

h = X*theta ;
diff = h - y ; 
theta1 = [0 ; theta(2:end ,:), :] ;
p = lambda*((theta1'*theta1)/(2*m)) ;
J = (diff'*diff)/(2*m) + p ;

grad = ( X'*diff + lambda*theta1 )/m ;






