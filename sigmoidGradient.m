function g = sigmoidGradient(z)
  % computes the gradient of the sigmoid function for matrix z

  s = sigmoid(z);
  g = s .* (1 - s);

end
