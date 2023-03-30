%% NIQE_check.m
% Cycles through the images in local folder demo_restored_images
% For each image, it will run the NIQUE (and get a Ma value too?)


%% NIQE

files       = dir('demo_restored_images/*.jpg');
niqe_array  = zeros(length(files));
niqe_array  = niqe_array(1,:);
i           = 1;

I               = imread('demo_restored_images/img_num_0_m_5.0.jpg');
niqe(I);

for file = files'
    I               = imread(file.name);
    %disp(I);
    niqe_array(i)   = niqe(I);
    i               = i + 1;
end

