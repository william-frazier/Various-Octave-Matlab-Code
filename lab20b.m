% CS 451 lab 20a - PCA, part 2

clear all; clf

%  Load Face dataset
load ('ex7faces.mat')
% 5000 faces, each 32 x 32 pixels, represented as a 1 x 1024 vectorize
% intensities are normalized to be between -1.0 and 1.0

%  Display the first 100 faces in the dataset
figure(1);
displayData(X(1:100, :));
title(' first 100 (of 5000) faces');

fprintf('Press enter to run PCA (will take a while...)\n');
pause;

[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);

%  Visualize the top 36 eigenvectors found
figure(2);
displayData(U(:, 1:36)');
title('top 36 eigenvectors');

% project data onto 100 dimensions
K = 100;
Z = projectData(X_norm, U, K);

fprintf('Original  data X has size of %d x %d\n', size(X));
fprintf('Projected data Z has size of %d x %d\n', size(Z));

% recover the data
X_rec  = recoverData(Z, U, K);

% Display original and reduced data side by side
figure(1);
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;
