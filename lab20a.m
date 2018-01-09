% CS 451 lab 20a - PCA, part 1

clear

load ('ex7data1.mat');   % loads variable X

fprintf('Visualizing example dataset for PCA.\n\n');
plot(X(:, 1), X(:, 2), 'bo');
axis([0 8 0 8]); axis square;

%%% now, implement pca.m

%return % comment out to proceed

%  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = featureNormalize(X);

fprintf('\nRunning PCA on example dataset.\n\n');
[U, S] = pca(X_norm);

% show top eigenvector:
u1 = U(:, 1)
% should be
%  -0.70711
%  -0.70711


%  Draw the eigenvectors centered at mean of data. These lines show the
%  directions and magnitudes of maximum variations in the dataset.
hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;

%%% now, implement projectData.m and recoverData.m

%return % comment out to proceed

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nDimension reduction on example dataset.\n\n');

%  Plot the normalized dataset (returned from pca)
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-3 3 -3 3]); axis square

%  Project the data onto K = 1 dimension
K = 1;
Z = projectData(X_norm, U, K);

% show first projected example
z1 = Z(1)
% should be 1.4813

X_rec  = recoverData(Z, U, K);

% show first recovered example
xrec1 = X_rec(1, :)
% should be   -1.0474  -1.0474


%  Draw lines connecting the projected points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off
