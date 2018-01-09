function checkNNGradients(lambda)
  % check backpropagation gradients using small test network

  s1 = 3;
  s2 = 5;
  s3 = 3;
  m = 5;

  % We generate some 'random' test data
  Theta1 = debugInitializeWeights(s2, s1);
  Theta2 = debugInitializeWeights(s3, s2);
  % Reusing debugInitializeWeights to generate X
  X  = debugInitializeWeights(m, s1 - 1);
  y  = 1 + mod(1:m, s3)';

  % Unroll parameters
  thetaVec = [Theta1(:); Theta2(:)];

  % Short hand for cost function
  costFunc = @(p) nnCostFunction(p, s1, s2, s3, X, y, lambda);

  [cost, grad] = costFunc(thetaVec);
  numgrad = computeNumericalGradient(costFunc, thetaVec);

  % Visually examine the two gradient computations.  The two columns
  % you get should be very similar. 
  disp([numgrad(1:8, :), grad(1:8, :)]);
  fprintf(['The above two columns should be very similar.\n' ...
           '(Left: numerical gradient, right: your analytical gradient)\n\n']);

  % Evaluate the norm of the difference between two solutions.  
  % If you have a correct implementation, and assuming EPSILON = 0.0001 
  % in computeNumericalGradient.m, then diff below should be less than 1e-9
  diff = norm(numgrad-grad)/norm(numgrad+grad);

  fprintf(['If your backpropagation implementation is correct, then \n' ...
           'the relative difference will be small (less than 1e-9). \n' ...
           '\nRelative Difference: %g\n'], diff);

end
