function h = forward (Theta1, Theta2, X)
% computes feedforward propagation of 3-layer neural net

  % layers have size s1, s2, s3
  % X has size m x s1
  % Theta1 has size s2 x (s1+1)
  % Theta2 has size s3 x (s2+1)
  % output h has size m x s3

  m = size(X, 1);

  % as in lab 10, but more compact (and we're transposing the Thetas!)

  a = sigmoid([ones(m, 1) X] * Theta1');
  h = sigmoid([ones(m, 1) a] * Theta2');

end
