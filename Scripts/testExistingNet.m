close all; clear all; clc;

load trainedDeeplabv3plus_RescueNet-256_256.mat; %!!!!!!!

outputFolder = fullfile('D:\','RescueNet/'); 

%Change the dataset accordingly
trainImgDir = fullfile(outputFolder,"MyTraining","train-256_256/"); %!!!!!!
imds = imageDatastore(trainImgDir);

%show an example image with its colormask and color bar
I = readimage(imds,3);
I = histeq(I);
I = imresize(I, [256 256]); %!!!!!!!
imshow(I)

%Load RescueNet Pixel-Labeled Images
labelIDs = rescueNetPixelLabelIDs();
classes = getClassNames();

labelDir = fullfile(outputFolder,"MyTraining","Colormasks-256_256/"); %!!!!!!
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

C = readimage(pxds,3);
cmap = rescueNetColorMap;
B = labeloverlay(I,C,ColorMap=cmap);
imshow(B); title("Example image with colormask and color bar");
pixelLabelColorbar(cmap,classes);

%Test Network on One Image
imgDir = fullfile(outputFolder,"MyTraining","test-256_256/");
imdsTest = imageDatastore(imgDir);

I = readimage(imdsTest,82);
C = semanticseg(I,net,Classes=classes); %segmentation
B = labeloverlay(I,C,Colormap=cmap,Transparency=0.4);

%get the name of test image
fullFileNames = vertcat(imdsTest.Files);
[folder, baseFileNameNoExtension, extension] = fileparts(fullFileNames{82}); %!!!!!
baseFileNameWithExtension = [baseFileNameNoExtension, extension];
%fprintf("Test file name #%d = %s\n", 6, baseFileNameWithExtension);

figure; %!!!!
imshow(B); title("\"+baseFileNameWithExtension+" and its segmentated colormask"); %raw image and segmentated colormask
pixelLabelColorbar(cmap, classes);

testLabelDir = fullfile(outputFolder,"MyTraining","test-Colormasks-256_256/"); %!!!!!!
pxdsTest = pixelLabelDatastore(testLabelDir,classes,labelIDs);

expectedResult = readimage(pxdsTest,82); %expected colormask
actual = uint8(C); %raw image and segmentated colormask
expected = uint8(expectedResult);
figure; %!!!!
imshowpair(actual, expected); title("Intersection");

iou = jaccard(C,expectedResult); %INTERSECTION OVER UNION
fprintf("Metrics for the test image:");
table(classes,iou)

%Evaluate Trained Network
pxdsResults = semanticseg(imdsTest,net, ...
    Classes=classes, ...
    MiniBatchSize=4, ...
    WriteLocation=tempdir, ...
    Verbose=false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,Verbose=false);
fprintf("Dataset Metrics:");
metrics.DataSetMetrics
fprintf("Class Metrics:");
metrics.ClassMetrics
fprintf("Confusion Matrix:");
metrics.NormalizedConfusionMatrix