%% NIQE_check.m
% Cycles through the images in local folder demo_restored_images
% For each image, it will run the NIQUE (and get a Ma value too?)

%% NIQE

path        = 'demo_restored_images/';
files       = dir([path '*.jpg']);
if (isempty(files))
    disp('There are no images in demo_restored_images/. Please run the demo.py file first.')
else
    niqe_array  = zeros(length(files));
    niqe_array  = niqe_array(1,:);
    brisque_arr = zeros(length(files));
    brisque_arr = brisque_arr(1,:);
    
    i           = 1;
    for file = files'
        address         = [path file.name];
        I               = imread(address);
        niqe_array(i)   = niqe(I);
        brisque_arr(i)   = brisque(I);
        i               = i + 1;
    end
end
disp('niqe scores:');
disp(niqe_array);
disp(' ');
disp('brisque scores:');
disp(brisque_arr);