function BagOfWord()

% Set training data
rootFolder = 'cifar10Train';
categories = {'Deer','Dog','Frog','Cat','Ship'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
if exist('Bow_imds.mat','file') == 2
    load('Bow_imds.mat');
else
    imds = splitEachLabel(imds, minSetCount, 'randomize');
    save('Bow_imds.mat','imds');
end

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
if exist('Bow_bag.mat','file') == 2
    load('Bow_bag.mat');
else
    bag = bagOfFeatures(imds);
    save('Bow_bag.mat','bag');
end

img = readimage(imds, 1);
featureVector = encode(bag, img);
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

% Train Data
categoryClassifier = trainImageCategoryClassifier(imds, bag);
% Test data
testFolder = 'cifar10Test';
imds = imageDatastore(fullfile(testFolder, categories), 'LabelSource', 'foldernames');

confMatrixTest = evaluate(categoryClassifier, imds);
mean(diag(confMatrixTest));


end

