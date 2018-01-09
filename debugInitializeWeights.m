function W = debugInitializeWeights(s2, s1)
  % create test matrix of size s2 x (1+s1) with fixed values for debugging

  W = zeros(s2, 1 + s1);
  % use sin function to obtain fixed values
  W = reshape(sin(1:numel(W)), size(W)) / 10;

end
