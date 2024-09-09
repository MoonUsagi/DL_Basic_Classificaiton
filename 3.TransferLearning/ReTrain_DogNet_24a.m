%% ReTrain_DogNet_24a
% Fred liu 2024.09.09

%% 初始化(Initialization)
%close all; clear all;clc

%% 輸入資料(Load Image Data)
folderName = 'Dog_Images';
DSdog = imageDatastore(folderName, ...
    IncludeSubfolders=true,...
    LabelSource='foldernames');
%% 分類名稱與種類(ClassName and Label)
classNames = categories(DSdog.Labels)
numClasses = numel(classNames)

%% 嵌入外部function > 正規化(normalization)
DSdog.ReadFcn = @preprocessImg;

%% 切割訓練與測試資料(Split Train & Test Data)
trainingNumFiles = 120;
[trainDigitData,testDigitData] = splitEachLabel(DSdog,trainingNumFiles, 'randomize');

%% 載入已訓練模型(Load Pretrained Network)
net = imagePretrainedNetwork("resnet18",NumClasses=numClasses);
%analyzeNetwork(net)

inputSize = networkInputSize(net);
[layerName,learnableNames] = networkHead(net)
net = freezeNetwork(net,LayerNamesToIgnore=layerName);

%% 訓練參數(Set Training Option)
options = trainingOptions("adam", ...
    InitialLearnRate=0.01,...
    MaxEpochs=30,...
    MiniBatchSize=64,...
    ValidationData=testDigitData, ...
    ValidationFrequency=5, ...
    Plots="training-progress", ...
    Metrics = ["accuracy","fscore"], ...
    Verbose=false);

%% Train Network
net = trainnet(trainDigitData,net,"crossentropy",options);

%% Test Network
YTest = minibatchpredict(net,testDigitData);
YTest = scores2label(YTest,classNames);

TTest = testDigitData.Labels;
figure,confusionchart(TTest,YTest);

acc = mean(TTest==YTest)