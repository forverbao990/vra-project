function DoAn()
tempdir = 'D:\vra-project\Data';
% Location of the compressed data set
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = fullfile(tempdir, 'caltech101');

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'airplanes', 'ferry', 'laptop','chair','cup','lotus'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
if exist('imds.mat','file') == 2
    load('imds.mat');
else
    imds = splitEachLabel(imds, minSetCount, 'randomize');
    save('imds.mat','imds');
end

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
if exist('bag.mat','file') == 2
    load('bag.mat');
else
    bag = bagOfFeatures(imds);
    save('bag.mat','bag');
end

img = readimage(imds, 1);
featureVector = encode(bag, img);
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

categoryClassifier = trainImageCategoryClassifier(imds, bag);

% Test data
outputTestDataFolder = fullfile(tempdir, 'TestData');
%rootTestDataFolder = fullfile(outputTestDataFolder, '101_ObjectCategories');
imds = imageDatastore(fullfile(outputTestDataFolder, categories), 'LabelSource', 'foldernames');
tbl01 = countEachLabel(imds)
confMatrixTest = evaluate(categoryClassifier, imds);
mean(diag(confMatrixTest));
%[trainingSet, validationSet] = splitEachLabel(imds, minSetCount, 'randomize');
%save('bag.mat','bag');
end

