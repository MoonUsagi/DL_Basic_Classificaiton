%% DL_5Lcode
% Fred liu 2022.5.19
%%
net = alexnet;

img = imread('2r.jpg');
figure,imshow(img)

img2 = imresize(img,[227 227]);
label = classify(net,img2)