% CS 451 lab 7b - Regularized Logistic Regression
% also find theta and plot decision boundary

% Name: William Frazier

clear; % clear variables
clf;   % clear figure

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

fprintf(['Plotting data with + indicating y=1 and o indicating y=0 examples.\n']);
plotData(X, y);

[m, n] = size(X);
X = [ones(m, 1) X];   % Add ones to X

fprintf('Running fminunc...\n');

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

lambda = 0;
initial_theta = zeros(n+1, 1);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y, lambda)), initial_theta, options)

% should be
% theta =
%  -25.16127
%    0.20623
%    0.20147
% cost =  0.20350

% plot the decision boundary
plotBoundary1(theta, X);

%return % comment out to proceed (implement predict.m first)

% Compute accuracy on our training set
p = predict(theta, X);
accuracy = mean(p == y)

% should be 0.89