function Iout= preprocessImg(filename)

I = imread(filename);

Iout = imresize(I, [227,227]);
end

