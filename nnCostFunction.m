function [J, grad] = nnCostFunction(thetaVec, s1, s2, s3, X, y, lambda)
% computes cost function and gradients for a three-layer neural network

  [Theta1, Theta2] = extractThetas(thetaVec, s1, s2, s3); % revert unrolling
  m = size(X, 1);
         
  %%%% 1. Cost function (vectorized)

  % feedforward computation (streamlined version of code from lab 10)

  h = forward(Theta1, Theta2, X);  % h has size m x s3

  % need to turn output vector y (which contains class labels) into boolean
  % matrix yy of same size as h (m x s3).  E.g. for m=4 and s3=3:
  % y =       yy =	    
  %           	    
  %    1          1   0   0
  %    3          0   0   1
  %    2          0   1   0
  %    3          0   0   1

  
  yy = y == (ones(m,1) * (1:s3));


  % cost function for neural nets



  jj = -yy .* log(h) - (1 - yy) .* log(1 - h);
  J = (1 / m) * sum(sum(jj));

  %%% (optional) add regularization
  %reg = ...
  %J += reg;


  %%%% 2. Backpropagation (using loop over training examples)

  Delta1 = zeros(size(Theta1));
  Delta2 = zeros(size(Theta2));

  for t = 1:m,
    xt = X(t, :)';
    yt = yy(t, :)';
    
    % 1. feed forward of single training example (from lab 10)
    a1 = [1; xt];
    z2 = Theta1*a1;
    a2 = [1;sigmoid(z2)];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);

    % 2. backprop
    % FILL IN
    delta3 = a3 - yt;
    delta2 = Theta2' * delta3;
    delta2 = delta2(2:end);
    delta2 = delta2 .* sigmoidGradient(z2);
    
    % 3. update Deltas
    % FILL IN
    Delta1 += delta2 * a1';
    Delta2 += delta3 * a2';

  end;

  Theta1_grad = (1 / m) * Delta1;
  Theta2_grad = (1 / m) * Delta2;

  %%% (optional) add regularization
  % ...
  %Theta1_grad += ...
  %Theta2_grad += ...


  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
