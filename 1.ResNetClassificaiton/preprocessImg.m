function Iout= preprocessImg(filename)

I= imread(filename);

Iout = imresize(I, [224,224]);

end

