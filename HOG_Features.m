function HOG_Features(rootFolder, testFolder, exportFolder)

%% Check data
%rootFolder = 'cifar10Train';
%testFolder = 'cifar10Test';

% Set training data
categories = {'Deer','Dog','Frog','Cat','Ship'};
trainingSet = imageDatastore(fullfile(rootFolder, categories),'IncludeSubfolders', true, 'LabelSource', 'foldernames');
trainingSet.ReadFcn = @readFunctionTrain;


testSet    = imageDatastore(fullfile(testFolder, categories), 'IncludeSubfolders', true,'LabelSource', 'foldernames');
testSet.ReadFcn = @readFunctionTrain;

countEachLabel(trainingSet)
countEachLabel(testSet)

img = readimage(trainingSet, 1);

% Extract HOG features and HOG visualization
%cellSize = [4 4];
%ellType = '4x4';

cellSize = [8 8];
cellType = '8x8';
[hog, ~] = extractHOGFeatures(img,'CellSize',cellSize);

hogFeatureSize = length(hog);

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

fprintf("\n1.Extract HOG Features from training set....");
if exist(strcat(exportFolder,'\trainingFeatures_',cellType,'.mat'),'file') == 2
    load(fullfile(exportFolder,strcat('trainingFeatures_',cellType,'.mat')));
else
    for i = 1:numImages
    img = readimage(trainingSet, i);

    img = rgb2gray(img);

    % Apply pre-processing steps
    img = imbinarize(img);
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
    end
    save(fullfile(exportFolder,strcat('trainingFeatures_',cellType,'.mat')),'trainingFeatures');
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;

if exist(strcat(exportFolder,'\classifier_',cellType,'.mat'),'file') == 2
    load(fullfile(exportFolder,strcat('classifier_',cellType,'.mat')));
    %load(strcat('classifier_',cellType,'.mat'));
else
    % fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
    classifier = fitcecoc(trainingFeatures, trainingLabels);
    %save(strcat('classifier_',cellType,'.mat'),'classifier');
    save(fullfile(exportFolder,strcat('classifier_',cellType,'.mat')),'classifier');
end

% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
fprintf("\n2.Extract HOG Features from test set....");
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);

% Make class predictions using the test features.
fprintf("\n3.Make class predictions....");
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
fprintf("\n4.Show result matrix....");
[confMat,order] = confusionmat(testLabels, predictedLabels);
fprintf('\n5.Display Confusion Matrix \n');
% Convert confusion matrix into percentage form
%confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
DisplayConfusionMatrix(confMat, order);

% Display the mean accuracy
%mean(diag(confMat))
actual = sum(predictedLabels==testLabels)/numel(predictedLabels) * 100;
fprintf('\n Actual = [%f]\n', actual);

end

