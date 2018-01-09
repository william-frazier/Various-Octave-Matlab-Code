function X_rec = recoverData(Z, U, K)
  % recover an approximation of the original data that was reduced
  % to K dimensions

  m = size(Z, 1);
  n = size(U, 1);
  U_reduce = U(:,1:K);

  % FILL IN
  
  % use a vectorized implementation similar to projectData
  % but using the transpose of Ureduced

  X_rec = Z * U_reduce';  % just a placeholder

end
