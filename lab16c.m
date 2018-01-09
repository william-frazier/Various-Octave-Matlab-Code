% CS 451 lab 16c - SVMs, part 3

% Name: William Frazier

clear;

% Load training data X, y
% and validation data Xval, yval
load('ex6data3.mat');

% Plot training data
%plotData(X, y);

C = 1;
sigma = 5;
% to try out different values interactively, uncomment the following:
%C = input("C:");
%sigma = input("sigma:");

fprintf('============ C = %.2f   sigma = %.2f\n', C, sigma);
best = 9999;
tol = 1e-3;
max_passes = 5;

for C=1:10:9
  for sigma=0.01:0.01:0.11
    fprintf('C=%.2f and sigma=%.2f', C, sigma);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma), ...
                    tol, max_passes);

    %figure(1);
    %visualizeBoundary(X, y, model);
    %title('training set');

    %figure(2);
    %visualizeBoundary(Xval, yval, model);
    %title('validation set');

    ptrain = svmPredict(model, X);
    errtrain = mean(ptrain ~= y);
    fprintf('training error   = %.2f%%\n', 100 * errtrain);

    pval = svmPredict(model, Xval);
    errval = mean(pval ~= yval);
    fprintf('validation error = %.2f%%\n', 100 * errval);
    if (errval < best);
      best = errval;
      record_c = C;
      record_sigma = sigma
      end
    end
end
best * 100
record_c
record_sigma
