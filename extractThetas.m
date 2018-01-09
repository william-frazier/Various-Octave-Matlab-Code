function [Theta1, Theta2] = extractThetas(thetaVec, s1, s2, s3)
% reverts unrolling of Theta matrices for a three-layer neural network

  n1 = s2 * (s1 + 1); % number of elements in Theta1
  Theta1 = reshape(thetaVec(1:n1),       s2, (s1 + 1));
  Theta2 = reshape(thetaVec((n1+1):end), s3, (s2 + 1));
end;
