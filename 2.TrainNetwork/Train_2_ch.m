%% �Ыؤp���`�׾ǲߺ����i���g�Ʀr����2
% (Create a small deep learning network for handwritten digit classification)
% Fred liu 2022.5.20

%% ���J�v�����(Load Image Data)
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% �q��Ʈw����ܼv��(Visualize Image from dataset)
figure;
perm = randperm(10000, 20);
for i = 1:20
    subplot(4,5,i);
    img = readimage(digitData, perm(i));
    imshow(img);
end

%% �T�{�C�Ӥ��������v���ƶq(Confirm the number of images in each category)
CountLabel = digitData.countEachLabel

%% ���ΰV�m�P���ո��(Split Train & Test Data)
trainingNumFiles = 750;
[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
    trainingNumFiles, 'randomize');

%% �w�q�����[�c(Define Network Architecture)
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,64,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% �]�w�V�m�Ѽ�(Set Training Option)
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.01,...
    'ExecutionEnvironment', 'auto',...
    'Plots', 'training-progress',...
    'ValidationData', testDigitData,...
    'ValidationFrequency', 30);

%% �V�m����(Train Network)
convnet = trainNetwork(trainDigitData, layers, options);

%% �b���ռv�����i��v������(Image Classification In Test Images)
predictedLabels  = classify(convnet, testDigitData);
valLabels  = testDigitData.Labels;

%% �p���ǫ�(Calculation Accuracy)
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

%% �p��V�c�x�}
figure
[cmat,classNames] = confusionmat(valLabels,predictedLabels);
h = heatmap(classNames,classNames,cmat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');