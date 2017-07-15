function DeepLearning(rootFolder, testFolder,exportFolder)

% Set training data
categories = {'Deer','Dog','Frog','Cat','Ship'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
%minSetCount = 50;

% Use splitEachLabel method to trim the set.
%if exist(fullfile('deep_imds.mat'),'file') == 2
if exist(strcat(exportFolder,'\deep_imds','.mat'),'file') == 2    
    %load(strcat('export', '/','deep_imds'), '-mat');
    load(fullfile(exportFolder,strcat('deep_imds','.mat')));
else
    imds = splitEachLabel(imds, minSetCount, 'randomize');
    %save('deep_imds.mat','imds');
    save(fullfile(exportFolder,strcat('deep_imds','.mat')),'imds');
end

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

%% Use alexnet library
net = alexnet();

% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readFunctionTrain(filename);

[trainingSet, ~] = splitEachLabel(imds, 0.3, 'randomize');

testSet    = imageDatastore(fullfile(testFolder, categories), 'IncludeSubfolders', true,'LabelSource', 'foldernames');
testSet.ReadFcn = @readFunctionTrain;

featureLayer = 'fc7'; %4096 fully connected layer
%featureLayer = 'fc8'; % 1000 fully connected layer
%featureLayer = 'conv4'; %Convolution: 384 3x3x192 convolutions with stride [1  1] and padding [1  1]

if exist(strcat(exportFolder,'\trainingFeatures_',featureLayer,'.mat'),'file') == 2 
    %load(strcat('export', '/','deep_imds'), '-mat');
    load(fullfile(exportFolder,strcat('trainingFeatures_',featureLayer,'.mat')));
else    
    trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
    %save('deep_imds.mat','imds');
    save(fullfile(exportFolder,strcat('trainingFeatures_',featureLayer,'.mat')),'trainingFeatures');
end

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
if exist(strcat(exportFolder,'\classifier_', featureLayer,'.mat'),'file') == 2 
    %load(strcat('export', '/','deep_imds'), '-mat');
    load(fullfile(exportFolder,strcat('classifier_',featureLayer,'.mat')));
else    
    classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
    %save('deep_imds.mat','imds');
    save(fullfile(exportFolder,strcat('classifier_', featureLayer,'.mat')),'classifier');
end

if exist(strcat(exportFolder,'\dpTestFeatures_', featureLayer,'.mat'),'file') == 2 
    %load(strcat('export', '/','deep_imds'), '-mat');
    load(fullfile(exportFolder,strcat('dpTestFeatures_',featureLayer,'.mat')));
else    
    % Extract test features using the CNN
    testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);
    save(fullfile(exportFolder,strcat('dpTestFeatures_', featureLayer,'.mat')),'testFeatures');
end

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
[confMat,order] = confusionmat(testLabels, predictedLabels);

% Display the mean accuracy
actual = sum(predictedLabels==testLabels)/numel(predictedLabels);
fprintf('\n Actual = [%f]\n', actual);
% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
DisplayConfusionMatrix(confMat, order);

end

