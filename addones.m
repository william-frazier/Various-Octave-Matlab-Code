# add column of ones to matrix M

function result = addones (M)
  [rows, cols] = size(M);
  b = ones(rows, 1);
  result = [b, M];
end
