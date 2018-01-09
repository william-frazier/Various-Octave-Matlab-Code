% CS 451 lab 11 - Neural Nets, cost function and backpropagation

% Name: William Frazier

%% Initialization
clear ; close all; clc

%%%%%% 1. Testing cost function

% neural net with 3 layers
s1 = 400; % 400 pixels as input (20x20 pixel images of handwritten digits)
s2 = 25;  % 25 hidden units
s3 = 10;  % 10 labels representing digits from 1 to 10
          % (note that we have mapped "0" to label 10)

% load training data X (5000 images) and y (5000 labels)
load('ex3data1.mat'); % (copy this file from your lab10 folder)
m = size(X, 1);

% if you want to see the data again, uncomment this:
% Randomly select 100 data points to display
%sel = randperm(m);
%sel = sel(1:100);
%displayData(X(sel, :));

% load pre-initialized neural network parameters Theta1 and Theta2
load('ex3weights.mat'); % (copy this file from your lab10 folder)

% Unroll parameters 
thetaVec = [Theta1(:) ; Theta2(:)];

lambda = 0; 
J0 = nnCostFunction(thetaVec, s1, s2, s3, X, y, lambda)
% should be 0.28763

% if you've implemented regularization, also check this:
%lambda = 1;
%J1 = nnCostFunction(thetaVec, s1, s2, s3, X, y, lambda)
% should be 0.38377

%return % comment out to proceed

%%%%%% 2. Testing Backprop gradients

% Gradient checking
lambda = 0;
checkNNGradients(lambda);

% if you've implemented regularization, also check this:
%lambda = 3;
%checkNNGradients(lambda);

%return % comment out to proceed

%%%%%% 3. Training the neural net

fprintf('\nTraining Neural Network... \n')

initial_Theta1 = randInitializeWeights(s1, s2);  % check out this function
initial_Theta2 = randInitializeWeights(s2, s3);

initial_thetaVec = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 50);  % if time, try larger (or smaller) values
lambda = 0; % if time and if you've implemented regularization, try other values

% Create cost function whose only argument is thetaVec
costFunction = @(p) nnCostFunction(p, s1, s2, s3, X, y, lambda);

% minimize it using fmincg (similar to fminunc, also utilizes gradient)

[thetaVec, cost] = fmincg(costFunction, initial_thetaVec, options);

[Theta1, Theta2] = extractThetas(thetaVec, s1, s2, s3);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Visualize what the neural network is learning by displaying the 
% hidden units to see what features they are capturing in the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

pred = predict(forward(Theta1, Theta2, X));

accuracy = 100 * mean(pred == y)  % prediction accuracy in percent

fprintf('Program paused. Press enter to continue.\n');
pause;

%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    fprintf('\nDisplaying Example Image\n');
    x = X(rp(i), :);
    displayData(x);
    pred = predict(forward(Theta1, Theta2, x));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end
