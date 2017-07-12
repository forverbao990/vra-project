function HOG_Features()
% Set training data
rootFolder = 'cifar10Train';
categories = {'Deer','Dog','Frog','Cat','Ship'};
trainingSet = imageDatastore(fullfile(rootFolder, categories),'IncludeSubfolders', true, 'LabelSource', 'foldernames');
trainingSet.ReadFcn = @readFunctionTrain;

testFolder = 'cifar10Test';
testSet    = imageDatastore(fullfile(testFolder, categories), 'IncludeSubfolders', true,'LabelSource', 'foldernames');
testSet.ReadFcn = @readFunctionTrain;

countEachLabel(trainingSet)
countEachLabel(testSet)

img = readimage(trainingSet, 1);

% Extract HOG features and HOG visualization
%[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);

cellSize = [4 4];
hogFeatureSize = length(hog_4x4);

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');
fprintf("\nExtract HOG Features from training set....");


if exist('trainingFeatures.mat','file') == 2
    load('trainingFeatures.mat');
else
    for i = 1:numImages
    img = readimage(trainingSet, i);

    img = rgb2gray(img);

    % Apply pre-processing steps
    img = imbinarize(img);
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
    end
    save('trainingFeatures.mat','trainingFeatures');
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;

if exist('classifier.mat','file') == 2
    load('classifier.mat');
else
    % fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
    classifier = fitcecoc(trainingFeatures, trainingLabels);
    save('classifier.mat','classifier');
end

% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
fprintf("\nExtract HOG Features from test set....");
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
fprintf("\nShow result matrix....");
[confMat,order] = confusionmat(testLabels, predictedLabels);
fprintf('\n');
% Convert confusion matrix into percentage form
%confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
DisplayConfusionMatrix(confMat, order)

% Display the mean accuracy
mean(diag(confMat))

end

