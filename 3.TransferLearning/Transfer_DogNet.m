%% Transfer_DogNet
% Fred liu 2022.5.20

%% 初始化(Initialization)
%close all; clear all;clc
%% 輸入資料(Load Image Data)
digitData = imageDatastore('Dog_Images', ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%% 正規化(normalization)
digitData.ReadFcn = @preprocessImg;
%% 切割訓練與測試資料(Split Train & Test Data)
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
%% 設定訓練參數(Set Training Option)
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 16,...
    'InitialLearnRate', 0.0001,...
    'ExecutionEnvironment', 'gpu',...
    'Plots', 'training-progress',...
    'ValidationData', testDigitData,...
    'ValidationFrequency', 30);
%% 訓練網路(Train Network)
[convnet,data] = trainNetwork(trainDigitData, layers_2, options);

%% 在測試影像中進行影像分類(Image Classification In Test Images)
predictedLabels  = classify(convnet, testDigitData);
valLabels  = testDigitData.Labels;

%% 計算精準度(Calculation Accuracy)
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

%% 計算混淆矩陣(Calculate the confusion matrix)
figure
[cmat,classNames] = confusionmat(valLabels,predictedLabels);
h = heatmap(classNames,classNames,cmat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');
