% CS 451 lab 16a - SVMs, part 1

% Name: William Frazier

clear;

% Load training data X, y
load('ex6data1.mat');

C = 1;
% to try out different values interactively, uncomment the following:
C = input("C:");

tol = 1e-5;
max_passes = 20;
model = svmTrain(X, y, C, @linearKernel, tol, max_passes);
visualizeBoundaryLinear(X, y, model);
