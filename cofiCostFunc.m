function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
xt = X * Theta';
d = R .* (xt - Y);
J = 0.5 * sum(sum(d .* d));
J += (lambda / 2) * sum(sum(Theta .* Theta));
J += (lambda / 2) * sum(sum(X .* X));

X_grad = zeros(size(X));
for i = 1:num_movies
  idx = find(R(i, :)==1);
  Thetatemp = Theta(idx, :);
  Ytemp = Y(i, idx);
    X_grad(i, :) = (X(i, :) * Thetatemp' - Ytemp) * Thetatemp;
end
X_grad += lambda * X;

Theta_grad = zeros(size(Theta));
for j = 1:num_users
  idx = find(R(:, j)==1);
  Xtemp = X(idx, :);
  Ytemp = Y(idx, j);
  d = (Xtemp * Theta(j, :)' - Ytemp);
  Theta_grad(j, :) = d' * Xtemp;
end
Theta_grad += lambda * Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
