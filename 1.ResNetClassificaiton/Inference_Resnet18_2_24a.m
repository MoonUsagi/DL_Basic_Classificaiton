%% New Classify Deep Learning Inference-2
% Fred liu 2024.09.09
% 2024a ~ new version

%% Load Pretrained Network

[net,classNames] = imagePretrainedNetwork("resnet18");
net.Layers

%% Create Datastore
DS_SimpleA = imageDatastore('SampleImages','IncludeSubfolders', true, 'LabelSource', 'foldernames');
DS_SimpleT = imageDatastore('1r.jpg','ReadFcn',@preprocessImg);

%% Count files in ImageDatastore labels
countEachLabel(DS_SimpleA)

%% Split New ImageDatastore 
DS_SimpleB = splitEachLabel(DS_SimpleA, 3);
countEachLabel(DS_SimpleB)

%% Change ReadFunction in imageDatastore 
DS_SimpleA.ReadFcn = @preprocessImg;
img = readimage(DS_SimpleA,5);

%% Classify and Display Image
scores = predict(net,single(img));
label = scores2label(scores,classNames);
figure,imshow(img)
title(string(label))




