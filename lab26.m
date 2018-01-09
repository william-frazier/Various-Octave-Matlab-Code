% CS 451 Lab 8 - collaborative filtering

% enter your user number here (see users.txt):
myId = 3;

Y = dlmread('rankings.txt');
R = (Y ~= 0);
[ids, titles] = loadTitles();

%fprintf('Average rating for movie 1 (%s): %f / 5\n\n', ...
%        titles{1}, mean(Y(1, R(1, :))));

my_ratings = Y(:, myId);
       
% visualize the ratings matrix (can comment out later)
%imagesc(Y);
%j = jet(18);
%colormap([0 0 0; j(3:17,:)]);
%ylabel('Movies');
%xlabel('Users');
%fprintf('Press enter to continue\n');
%pause;

% run collaborative filtering algorithm:

num_users = size(Y, 2);
num_movies = size(Y, 1);

% *** try varying these parameters:
num_features = 100
lambda = 1

% normalize ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

% random initialization
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% optimize
options = optimset('GradObj', 'on', 'MaxIter', 100);
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                      num_features, lambda)), initial_parameters, options);

% Unfold the returned theta
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

% make predictions
p = X * Theta';
my_predictions = p(:, myId) + Ymean;

if (1) % see original vs predicted ratings
  fprintf('\nOriginal vs. predicted ratings:\n');
  for i = 1:num_movies
    if my_ratings(i) > 0 
        fprintf('Rated %d, predicted %.1f, for: %s\n', ...
           my_ratings(i), my_predictions(i), titles{i});
    end
  end
  fprintf("Rated 5, predicted %.1f, for: Goodfellas\n", my_predictions(33));
  fprintf("Rated 4, predicted %.1f, for: Heat\n", my_predictions(36));
  fprintf("Rated 4, predicted %.1f, for: Silence of the Lambs\n", my_predictions(60));
endif

if (1) % see all predictions
  fprintf('\nAll predictions:\n');
  for i=1:num_movies
    fprintf('%d: Predicting rating %.1f for %s\n', i, my_predictions(i), titles{i});
  end
endif

if (1) % see recommendations
  [r, ix] = sort(my_predictions, 'descend');
  fprintf('\nTop 10 recommendations for you:\n');
  for i=1:10
    j = ix(i);
    fprintf('%d: Predicting rating %.1f for %s\n', i, my_predictions(j), titles{j});
  end

  fprintf('\nBottom 10 recommendations for you:\n');
  for i=num_movies-9:num_movies
    j = ix(i);
    fprintf('%d: Predicting rating %.1f for %s\n', i, my_predictions(j), titles{j});
  end
endif
