I1          = imread('Cat2.arw');
I2          = imread('Cat3.arw');
I3          = imread('img2.ARW');
I4          = imread('img3.jpg');
niqe1       = niqe(I1);
brisque1    = brisque(I1);
sprintf('Cat 2:\n  niqe = %f\n brisque = %f', niqe1, brisque1)
niqe2       = niqe(I2);
brisque2    = brisque(I2);
sprintf('Cat 3:\n  niqe = %f\n brisque = %f', niqe2, brisque2)
niqe3       = niqe(I3);
brisque3    = brisque(I3);
sprintf('Img2:\n  niqe = %f\n brisque = %f', niqe3, brisque3)
niqe4       = niqe(I4);
brisque4    = brisque(I4);
sprintf('Img3:\n  niqe = %f\n brisque = %f', niqe4, brisque4)
