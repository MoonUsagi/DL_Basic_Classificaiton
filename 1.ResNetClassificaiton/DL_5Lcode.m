%% DL_5Lcode
% Fred liu 2022.5.19
% update 2023.05.17
%%
net = resnet18;

img = imread('2r.jpg');
figure,imshow(img)

img2 = imresize(img,[224 224]);
label = classify(net,img2)