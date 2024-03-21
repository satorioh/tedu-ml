% .
% ├── annotations.mat
% ├── demo.m
% ├── formatted_annotations
% │   ├── 20151127_114556.txt
% │   ├── 20151127_114946.txt
% │   ├── 20151127_115133.txt
% │   ├── ...
% │   └── 20151221_135642.txt
% └── load_annotations.m

%% load_annotations.m

clc; clear;

% output path
output = './formatted_annotations/';

% Load the annotations in a map structure
load('annotations.mat');

% Each entry in the map corresponds to the annotations of an image.
% Each entry contains many cell tuples as annotated food
% A tuple is composed of 8 cells with the annotated:
% - (1) item category (food for all tuples)
% - (2) item class (e.g. pasta, patate, ...)
% - (3) item name
% - (4) boundary type (polygonal for all tuples)
% - (5) item's boundary points [x1,y1,x2,y2,...,xn,yn]
% - (6) item's bounding box [x1,y1,x2,y2,x3,y3,x4,y4]

image_names = annotations.keys;

n_images = numel(image_names);

for j = 1 : n_images

    image_name = image_names{j};
    tuples = annotations(image_name);
    count = size(tuples,1);
    coordinate_mat = cell2mat(tuples(:,6));

    % open file
    file_path = [output image_name '.txt'];
    ffile = fopen(file_path, 'w');

    % write file
    for k = 1 : count
        item = tuples(k,:);
        fprintf(ffile, '%s %d %d %d %d %d %d %d %d\n', ...
            string(item(2)), ...  % item class
            coordinate_mat(k,:)); % item's bounding box
    end

    % close file
    fclose(ffile);

end

%% fprintf
% Write data to text file
% https://www.mathworks.com/help/matlab/ref/fprintf.html