clear; clc;

% Step 1: Load and prepare dataset
imageSize = [256 256]; % The minimum image size is [224 224] because an FCN is based on the VGG-16 network.

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

% Step 2: Create FCN layers
numClasses = numel(classes);

lgraph = fcnLayers(imageSize, numClasses); % FCN is preinitialized using layers and weights from the vgg16 network. The model is also configured as FCN-8s.
%FCN-8s provides finer-grain segmentation at the cost of additional computation.

% Visualize the FCN architecture
plot(lgraph);

% Step 3: Set training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 50, ...
    'LearnRateSchedule', 'piecewise', ...
    'ExecutionEnvironment', 'gpu');

% Step 4: Train the FCN model
[net, info] = trainNetwork(trainingData, lgraph, options);

% Step 5: Save the trained network
save('trainedFCN_RescueNet.mat', 'net');