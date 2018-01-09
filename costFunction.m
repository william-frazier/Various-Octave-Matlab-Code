function [J, grad] = costFunction(theta, X, y, lambda)
%costFunction Compute cost and gradient for logistic regression with regularization
%   J = costFunction(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================

% initially (for problem 3) ignore the parameter lambda
c = (-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)));  
scalar = lambda/(2*m);
theta2 = theta;
theta2(1) = 0;
theta3 = theta2'*theta2;
reg = scalar * theta3;
J = mean(c) + reg;
d = sigmoid(X*theta)-y;
grad2 = (X'*d)/m;
grad = grad2 + (lambda/m) * theta2;

% Note: grad should have the same dimensions as theta



% =============================================================

end
