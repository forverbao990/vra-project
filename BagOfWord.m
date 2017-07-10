function BagOfWord()

% Set training data
rootFolder = 'cifar10Train';
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

% img = readimage(imds, 1);
% featureVector = encode(bag, img);
% figure
% bar(featureVector)
% title('Visual word occurrences')
% xlabel('Visual word index')
% ylabel('Frequency of occurrence')

% Train Data
fprintf("train Image Category Classifier....");
categoryClassifier = trainImageCategoryClassifier(imds, bag);
% Test data
testFolder = 'cifar10Test';
imds = imageDatastore(fullfile(testFolder, categories), 'LabelSource', 'foldernames');

fprintf("Evaluate matrix....");
confMatrixTest = evaluate(categoryClassifier, imds);
mean(diag(confMatrixTest));


end

