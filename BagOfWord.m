function BagOfWord(rootFolder, testFolder, exportFolder)

% Set training data
%rootFolder = 'cifar10Train';
Type = 'BoW';
categories = {'Deer','Dog','Frog','Cat','Ship'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category = 5000
%minSetCount = 50;

% Use splitEachLabel method to trim the set.
%if exist('minBow_imds.mat','file') == 2
if exist(strcat(exportFolder,'\imds',Type,'.mat'),'file') == 2
    load(fullfile(exportFolder,strcat('imds',Type,'.mat')));
else
    imds = splitEachLabel(imds, minSetCount, 'randomize');
    %save('minBow_imds.mat','imds');
    save(fullfile(exportFolder,strcat('imds',Type,'.mat')),'imds');
end

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
%if exist('minBow_bag.mat','file') == 2
if exist(strcat(exportFolder,'\bag',Type,'.mat'),'file') == 2
    load(fullfile(exportFolder,strcat('bag',Type,'.mat')));
else
    fprintf("Build Bag Of Features....");
    bag = bagOfFeatures(imds);
    save(fullfile(exportFolder,strcat('bag',Type,'.mat')),'bag');
end

% Train Data
fprintf("train Image Category Classifier....");
categoryClassifier = trainImageCategoryClassifier(imds, bag);
% Test data
imds = imageDatastore(fullfile(testFolder, categories), 'LabelSource', 'foldernames');

fprintf("Evaluate matrix....");
evaluate(categoryClassifier, imds);
%mean(diag(confMatrixTest))

end

