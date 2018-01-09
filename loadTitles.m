function [ids, titles] = loadTitles()
  fid = fopen('movietitles.txt');
  n = 80;
  ids = zeros(n, 1);
  titles = cell(n, 1);
  for i = 1:n
    line = fgets(fid);
    [id, title] = strtok(line);
    ids(i) = str2double(id);
    titles{i} = strtrim(title);
  end
  fclose(fid);
end
