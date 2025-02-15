clear; clc;

% Step 1: Load and prepare dataset
imageSize = [256 256 3];

classes = getClassNames();

labelIDs = rescueNetPixelLabelIDs();


outputFolder = fullfile('D:\','RescueNet/'); 
trainImgDir = fullfile(outputFolder,"MyTraining","train-256_256/");
labelDir = fullfile(outputFolder,"MyTraining","Colormasks-256_256/");
validationDir = fullfile(outputFolder,"MyTraining","val-256_256");
validationLabelDir = fullfile(outputFolder,"MyTraining","val-Colormasks-256_256");
testDir = fullfile(outputFolder,"MyTraining","test-256_256");
testLabelDir = fullfile(outputFolder,"MyTraining","test-Colormasks-256_256");

% Create image datastores for training and validation
trainImages = imageDatastore(trainImgDir);
trainLabels = pixelLabelDatastore(labelDir, classes, labelIDs);

valImages = imageDatastore(validationDir);
valLabels = pixelLabelDatastore(validationLabelDir, classes, labelIDs);

testImages = imageDatastore(testDir);
testLabels = pixelLabelDatastore(testLabelDir, classes, labelIDs);

% Combine image and label datastores
trainingData = pixelLabelImageDatastore(trainImages, trainLabels);
validationData = pixelLabelImageDatastore(valImages, valLabels);
testData = pixelLabelImageDatastore(testImages, testLabels);

% Step 2: Create SegNet layers
numClasses = numel(classes);
lgraph = segnetLayers(imageSize, numClasses, 'vgg16'); % Use VGG-16 as encoder

% Visualize the SegNet architecture
plot(lgraph);

% Step 3: Set training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 4, ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 50, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 5, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationPatience', 4);

% Step 4: Train the SegNet model
[net, info] = trainNetwork(trainingData, lgraph, options);

% Step 5: Save the trained network
save('trainedSegNet_RescueNet.mat', 'net');
