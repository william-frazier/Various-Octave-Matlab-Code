% CS 451 lab 7 - Regularized Logistic Regression

% name:

% 1. getting started

%clear ; close all; clc

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

%fprintf(['Plotting data with + indicating y=1 and o indicating y=0 examples.\n']);
%plotData(X, y);

% you may want to comment out the above two lines before proceeding

%return % comment out to proceed

% 2. sigmoid function

t = [-6 -1; 0 2]
st = sigmoid(t)

% should be 
% st =
%   0.0024726   0.2689414
%   0.5000000   0.8807971

%return % comment out to proceed

% 3. cost function w/o regularization

[m, n] = size(X);
X = [ones(m, 1) X];   % Add ones to X

lambda = 0;
theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(theta, X, y, lambda)

% should be
% cost =  0.21833
% grad =
%    0.042903
%    2.566234
%    2.646797

%return % comment out to proceed

% 4. cost function with regularization

% after you add regularization, the previous output
% should not change since lambda = 0 above

lambda = 15;  % now try a non-zero lambda
[cost, grad] = costFunction(theta, X, y, lambda)

% should  be
% cost =  0.22433
% grad =
%    0.042903
%    2.596234
%    2.676797
