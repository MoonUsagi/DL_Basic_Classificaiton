%% DL_5Lcode
% Fred liu 2022.5.19

%% Use ResNet18 to do inference
% Load Pre-trained CNN
net = resnet18;
% Show the architecture of ResNet18
net.Layers
%% Classify 
% Import a testing image.
img = imread('2r.jpg');

% There is a size requirement of 224 x 224 for ResNet18. 
img = imresize(img, [224 224]);

% Recognize the testing image
[Ypred, scores] = classify(net, img);

% Show predicting result
imshow(img);
title(char(Ypred))

%% List top 3 class scores
[ssort, sidx] = sort(scores, 'descend');

numTopClasses = 3; % show top N choices
TopClasses = net.Layers(end).ClassNames(sidx(1:numTopClasses));
TopScores = ssort(1:numTopClasses)';

topTable = table(TopClasses, TopScores)

