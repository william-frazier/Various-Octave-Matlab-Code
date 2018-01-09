% CS 451 lab 16b - SVMs, part 2

% Name: William Frazier

% fill in gaussianKernel.m

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma)
% should be 0.324652

%return % comment out to proceed

clear;

% Load training data X, y
load('ex6data2.mat');

plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

% SVM Parameters
C = 1; sigma = 0.1;

% limit tolerance and max passes so it doesn't run forever...
tol = 1e-3;
max_passes = 5;
    
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma), ...
                tol, max_passes);

visualizeBoundary(X, y, model);
