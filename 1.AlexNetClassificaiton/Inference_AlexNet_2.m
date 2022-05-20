clear all,close all,clc
%% ImageDatastore
net = alexnet;
net.Layers

%% Show what AlexNet does with random images without being retrained
samples = imageDatastore('SampleImages',...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Count files in ImageDatastore labels
countEachLabel(samples)

%% Split ImageDatastore labels by proportions
samplespart = splitEachLabel(samples, 3);
countEachLabel(samplespart)
 
%% Change ReadFunction in imageDatastore 
% Resize image before reading it
samplespart.ReadFcn = @preprocessImg;
img = readimage(samplespart,5);
%whos img

% Make prediction
classLabel = classify(net, img);

% Show image
imshow(img); 
title(char(classLabel));
