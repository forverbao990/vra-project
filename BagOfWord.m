function BagOfWord()

%% Check data
rootFolder = 'cifar10Train';
testFolder = 'cifar10Test';

if exist(rootFolder,'dir') ~= 7    
    fprintf("\nNo data train, please run DownloadCIFAR10 file... \n");
    return;
end

if exist(testFolder,'dir') ~= 7
    fprintf("\nNo data Test, please run DownloadCIFAR10 file... \n");
    return;
end

if exist('cifar-10-batches-mat','dir') ~= 7
    fprintf("\nNo cifar-10-batches-mat folder , please run DownloadCIFAR10 file... \n");
    return;
end  

% Set training data
%rootFolder = 'cifar10Train';
categories = {'Deer','Dog','Frog','Cat','Ship'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category = 5000
%minSetCount = 50;

% Use splitEachLabel method to trim the set.
if exist('minBow_imds.mat','file') == 2
    load('minBow_imds.mat');
else
    imds = splitEachLabel(imds, minSetCount, 'randomize');
    save('minBow_imds.mat','imds');
end

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
if exist('minBow_bag.mat','file') == 2
    load('minBow_bag.mat');
else
    fprintf("Build Bag Of Features....");
    bag = bagOfFeatures(imds);
    save('minBow_bag.mat','bag');
end

% Train Data
fprintf("train Image Category Classifier....");
categoryClassifier = trainImageCategoryClassifier(imds, bag);
% Test data
%testFolder = 'cifar10Test';
imds = imageDatastore(fullfile(testFolder, categories), 'LabelSource', 'foldernames');

fprintf("Evaluate matrix....");
confMatrixTest = evaluate(categoryClassifier, imds);
mean(diag(confMatrixTest));

end

