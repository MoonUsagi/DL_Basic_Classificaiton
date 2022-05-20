%% PreprocessImg
% Fred liu 2022.5.20

%%
function Iout= preprocessImg(filename)

I = imread(filename);

Iout = imresize(I, [227,227]);
end

