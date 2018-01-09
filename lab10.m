% CS 451 lab 10 - Neural Nets, feedforward propagation

% Name: William Frazier

%% Initialization
clear ; close all; clc

%%%%%% 1. simple test case

% neural net with 3 layers
s1 = 3;  % number of units in layer 1 (inputs / features) 
s2 = 3;  % number of units in layer 2 (units in hidden layer)
s3 = 2;  % number of units in layer 3 (outputs / classes)

% input matrix, size m x s1
X = [
1 2 4;
2 1 0;
3 4 5;
1 1 2;
2 6 3;
4 5 5];
m = size(X, 1);

% matrix with weights between layers 1 and 2, size s2 x (s1+1)
Theta1 = [
 .1 -.2  .4  .1;
-.3  .1 -.3 -.6;
 .2  .5 -.3  .2];

% matrix with weights between layers 2 and 3, size s3 x (s2+1)
Theta2 = [
 .1 -.4 -.1  .3;
-.1  .2  .5 -.1];

x = X(1, :)'  % first training example
% add feedforward steps here
a1 = [1;x];
z2 = Theta1*a1
a2 = [1;sigmoid(z2)]
z3 = Theta2*a2;
h = sigmoid(z3)

% should get h = [0.50232; 0.49963]

%return % comment out to proceed

%%%%%% 2. Vectorized feedforward

% now implement the forward function
h = forward(Theta1, Theta2, X)
% should get
% h =
%   0.50232   0.49963
%   0.51579   0.53359
%   0.50275   0.49801
%   0.51064   0.50818
%   0.47025   0.51136
%   0.50230   0.49829

p = predict(h) % find location of row-wise maximum 
% should get
% p =
%   1
%   2
%   1
%   1
%   2
%   1

%return % comment out to proceed

%%%%%% 3. Recognizing handwritten digits

% now for a bigger problem!  nothing to code, just watch!

% neural net with 3 layers
s1 = 400; % 400 pixels as input (20x20 pixel images of handwritten digits)
s2 = 25;  % 25 hidden units
s3 = 10;  % 10 labels representing digits from 1 to 10
          % (note that we have mapped "0" to label 10)

% load training data X (5000 images) and y (5000 labels)
load('ex3data1.mat');
m = size(X, 1);

% load pre-initialized neural network parameters Theta1 and Theta2
load('ex3weights.mat');

% Randomly select 100 data points to display
sel = randperm(m);
sel = sel(1:100);
displayData(X(sel, :));

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
