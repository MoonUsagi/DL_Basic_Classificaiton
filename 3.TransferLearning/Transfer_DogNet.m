%% Transfer_DogNet
% Fred liu 2022.5.20

%% ��l��(Initialization)
%close all; clear all;clc
%% ��J���(Load Image Data)
digitData = imageDatastore('Dog_Images', ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%% ���W��(normalization)
digitData.ReadFcn = @preprocessImg;
%% ���ΰV�m�P���ո��(Split Train & Test Data)
trainingNumFiles = 120;

[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
trainingNumFiles, 'randomize');

%% Model
Net = alexnet;
%Net = vgg16();
% NewNet = Net.Layers(1:end-3);
% 
% Layer = [NewNet
%     fullyConnectedLayer(3,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%     softmaxLayer
%     classificationLayer
%      ];
%% �]�w�V�m�Ѽ�(Set Training Option)
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 16,...
    'InitialLearnRate', 0.0001,...
    'ExecutionEnvironment', 'gpu',...
    'Plots', 'training-progress',...
    'ValidationData', testDigitData,...
    'ValidationFrequency', 30);
%% �V�m����(Train Network)
[convnet,data] = trainNetwork(trainDigitData, layers_2, options);

%% �b���ռv�����i��v������(Image Classification In Test Images)
predictedLabels  = classify(convnet, testDigitData);
valLabels  = testDigitData.Labels;

%% �p���ǫ�(Calculation Accuracy)
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

%% �p��V�c�x�}(Calculate the confusion matrix)
figure
[cmat,classNames] = confusionmat(valLabels,predictedLabels);
h = heatmap(classNames,classNames,cmat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');
