% CS 451 lab 8 - Regularized Logistic Regression, cont'd

% Name: William Frazier

clear; % clear variables
clf;   % clear figure

data = load('ex2data2.txt');  % micro chip quality testing data

X = data(:, [1, 2]); y = data(:, 3);
plotData2(X, y);

%  You are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).

degree = 6;   % max degree of polynomial
lambda = 1;   % regularization parameter

% to try out different values interactively, uncomment the following:
degree = input("degree:");
lambda = input("lambda:");

X = mapFeature(X(:,1), X(:,2), degree);  % add polynomial features
% mapFeature also adds a column of ones for us

[m, n] = size(X)  % here n includes the column of ones

test_theta = (1:n)' / (n+1);
[cost, grad] = costFunction(test_theta, X, y, lambda);

% verify cost and gradient (for degree=6, lambda=1)
%cost
% should be 1.0825

%grad(1:3)'  % print first three gradient values
% should be  0.142085   0.060779   0.077584

% comment out the above once verified correct

%return % comment out to proceed

initial_theta = zeros(n, 1);
options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunction(t, X, y, lambda)), initial_theta, options);

plotBoundary2(theta, degree, lambda);

% Compute accuracy on our training set
p = predict(theta, X);
accuracy = mean(p == y)

% should be 0.83051 (for degree=6, lambda=1)
