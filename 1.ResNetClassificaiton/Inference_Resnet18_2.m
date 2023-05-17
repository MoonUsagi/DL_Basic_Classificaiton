clear all,close all,clc
%% ImageDatastore
net = resnet18;
net.Layers
 
%% Show what ResNet18 does with random images without being retrained
samples = imageDatastore('SampleImages',...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Count files in ImageDatastore labels
countEachLabel(samples)

%% Split ImageDatastore labels by proportions
samplespart = splitEachLabel(samples, 3);
countEachLabel(samplespart)
 
%% Change ReadFunction in imageDatastore 
% Resize image before reading it
samples.ReadFcn = @preprocessImg;
img = readimage(samples,5);
%whos img

% Make prediction
classLabel = classify(net, img);

% Show image
imshow(img); 
title(char(classLabel));
