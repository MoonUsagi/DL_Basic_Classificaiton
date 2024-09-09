%% 創建小型深度學習網路進行手寫數字分類2
% (Create a small deep learning network for handwritten digit classification)
% 2024.05.22 for 24a update

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
net = dlnetwork;

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
];

net = addLayers(net,layers);
%% 設定訓練參數(Set Training Option)
options = trainingOptions(...
    'sgdm',...
    MaxEpochs = 10, ...
    Metrics = ["accuracy","fscore","recall"], ...
    ObjectiveMetricName="recall", ...
    MiniBatchSize = 128,...
    InitialLearnRate = 0.01,...
    ValidationData = testDigitData,...
    ValidationFrequency=35, ...
    ExecutionEnvironment = 'auto',...
    Plots = 'training-progress');
% Supported values are "accuracy", "auc", "fscore", "precision", 
% "recall", and "rmse".

%% 損失函數(loss function)
% crossentropy,binary-crossentropy,mse,huber
lossFcn = 'crossentropy';

%% 訓練網路(Train Network)
netTrained = trainnet(trainDigitData,net,lossFcn,options);

%% 在測試影像中進行影像分類(Image Classification In Test Images)
scores = minibatchpredict(netTrained,testDigitData);
classNames = categories(trainDigitData.Labels)
YValidation = scores2label(scores,classNames);

%%
TValidation = testDigitData.Labels;
accuracy = mean(YValidation == TValidation)