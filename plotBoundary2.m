function plotBoundary2(theta, degree, lambda)
%plotBoundary2 plots the nonlinear decision boundary defined by theta

hold on

% Here is the grid range
u = linspace(-1, 1.5, 50);
v = linspace(-1, 1.5, 50);

z = zeros(length(u), length(v));
% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = mapFeature(u(i), v(j), degree)*theta;
    end
end
z = z'; % important to transpose z before calling contour

% Plot z = 0
% Notice you need to specify the range [0, 0]
contour(u, v, z, [0, 0], 'LineWidth', 2)
title(sprintf('decision boundary, lambda = %g', lambda))

hold off

end
