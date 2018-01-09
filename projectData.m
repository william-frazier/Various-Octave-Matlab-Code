function Z = projectData(X, U, K)
  % compute the projection of the normalized inputs onto the 
  % reduced dimensional space spanned by the first K columns of U

  [m, n] = size(X);

  % FILL IN
  U_reduce = U(:, 1:K);
  % again, use a vectorized implementation
  % both X and Z have m rows, one per data point
  % extract the first K columns of U into Ureduced
  % then multiply each x(i) with Ureduced to get z(i)

  Z = X*U_reduce;  % just a placeholder

end
