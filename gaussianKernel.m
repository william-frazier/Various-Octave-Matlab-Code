function sim = gaussianKernel(x1, x2, sigma)
% computes the Gaussian kernel between x1 and x2

  % Ensure that x1 and x2 are column vectors
  x1 = x1(:); x2 = x2(:);
  diff = (x1 - x2).^2;
  sum = sum(diff);
  denom = 2 * sigma^2;

  sim = exp(-sum/denom); % just a placeholder
    
end
