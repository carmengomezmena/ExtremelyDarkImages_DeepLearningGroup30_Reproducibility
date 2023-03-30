%% NIQE_check.m
% Cycles through the images in local folder demo_restored_images
% For each image, it will run the NIQUE (and get a Ma value too?)


%% NIQE

path        = 'demo_restored_images/';
files       = dir([path '*.jpg']);
niqe_array  = zeros(length(files));
niqe_array  = niqe_array(1,:);

i           = 1;
for file = files'
    address         = [path file.name];
    I               = imread(address);
    niqe_array(i)   = niqe(I);
    i               = i + 1;
end
