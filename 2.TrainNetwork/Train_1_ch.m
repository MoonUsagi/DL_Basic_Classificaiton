%% 創建小型深度學習網路進行手寫數字分類
% (Create a small deep learning network for handwritten digit classification)
% Fred liu 2022.5.20

%% 載入影像資料(Load Image Data)
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% 從資料庫中顯示影像(Visualize Image from dataset)
figure;
perm = randperm(10000, 20);
for i = 1:20
    subplot(4,5,i);
    img = readimage(digitData, perm(i));
    imshow(img);
end

%% 確認每個分類中的影像數量(Confirm the number of images in each category)
CountLabel = digitData.countEachLabel

%% 切割訓練與測試資料(Split Train & Test Data)
trainingNumFiles = 750;
[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
    trainingNumFiles, 'randomize');

%% 定義網路架構(Define Network Architecture)
% 這邊請用Deepnetwork design拉出一個與下方一樣的模型
% layers = [
%     imageInputLayer([28 28 1])
%     convolution2dLayer(5, 20)
%     reluLayer
%     maxPooling2dLayer(2, 'Stride', 2)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

%% 設定訓練參數(Set Training Option)
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.01,...
    'ExecutionEnvironment', 'auto',...
    'Plots', 'training-progress');

%% 訓練網路(Train Network)
convnet = trainNetwork(trainDigitData, layers_1, options);

%% 在測試影像中進行影像分類(Image Classification In Test Images)
predictedLabels  = classify(convnet, testDigitData);
valLabels  = testDigitData.Labels;

%% 計算精準度(Calculation Accuracy)
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)
