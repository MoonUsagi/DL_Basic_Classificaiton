%% New Classify Deep Learning Inference - 1
% Fred liu 2024.09.09
% 2024a ~ new version

%% Load Pretrained Network

[net,classNames] = imagePretrainedNetwork("resnet18");

%% Read and Resize Image

img = imread("1r.jpg");
inputSize = net.Layers(1).InputSize;
img2 = imresize(img,inputSize(1:2));

%% Classify and Display Image

scores = predict(net,single(img2));
label = scores2label(scores,classNames);
figure,imshow(img2)
title(string(label))