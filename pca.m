function [U, S] = pca(X)
  % performs PCA: computes eigenvectors of the covariance matrix of X,
  % an m x n matrix of all m data points
  % returns the eigenvectors U and the eigenvalues (on diagonal) in S

  [m, n] = size(X);

  % FILL IN
  
  Sigma = 1/m * X' * X;
  [U, S, V] = svd(Sigma);
  % compute covariance matrix (one line in vectorized form, see
  % PCA algorithm video, about 2 minutes from the end)
  
  % then, call the "svd" function on the covariance matrix to compute
  % U, S, V  (only the first two will be returned from this function).
  
  %U = zeros(n);  % just placeholders
  %S = zeros(n);

end
