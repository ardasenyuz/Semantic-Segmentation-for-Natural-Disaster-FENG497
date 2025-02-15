outputFolder = fullfile('D:\','RescueNet\');

parentFile = fullfile(outputFolder,"train","train-org-img\");

resizeFile = fullfile(outputFolder,"MyTraining","train-216_216\");

images = dir(parentFile);

dirFlags = [images.isdir];

imageList = images(~dirFlags);

imagesNames = {imageList.name};

for K = 1 : length(imagesNames)
    imgPath = parentFile + "\" + imagesNames{K};
    I = imread(imgPath);
    Iresize = imresize(I,[216 216]);
    imwrite(Iresize, resizeFile+"_"+imagesNames{K});
end

