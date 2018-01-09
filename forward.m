function h = forward (Theta1, Theta2, X)
% computes feedforward propagation of 3-layer neural net

  % layers have size s1, s2, s3
  % X has size m x s1
  % Theta1 has size s2 x (s1+1)
  % Theta2 has size s3 x (s2+1)
  % output h has size m x s3

  m = size(X, 1);
  a1 = addones(X);
  z2 = Theta1 * a1';
  a2 = sigmoid(z2');
  a2 = addones(a2);
  z3 = Theta2 * a2';
  
  
  % placeholder
  h = sigmoid(z3');

end
