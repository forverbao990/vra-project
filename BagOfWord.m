function BagOfWord(rootFolder, testFolder, exportFolder, Type)

% Set training data
%rootFolder = 'cifar10Train';

categories = {'Deer','Dog','Frog','Cat','Ship'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;

%tbl = countEachLabel(imds);
%minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category = 5000
%minSetCount = 50;

% Use splitEachLabel method to trim the set.
%if exist('minBow_imds.mat','file') == 2
% if exist(strcat(exportFolder,'\imds',Type,'.mat'),'file') == 2
%     load(fullfile(exportFolder,strcat('imds',Type,'.mat')));
% else
%     imds = splitEachLabel(imds, minSetCount, 'randomize');
%     %save('minBow_imds.mat','imds');
%     save(fullfile(exportFolder,strcat('imds',Type,'.mat')),'imds');
% end

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
%if exist('minBow_bag.mat','file') == 2
if exist(strcat(exportFolder,'\bag',Type,'.mat'),'file') == 2
    load(fullfile(exportFolder,strcat('bag',Type,'.mat')));
else
    fprintf("Build Bag Of Features....");
    if  strcmp(Type,'CustomBow') ~= 1
        bag = bagOfFeatures(imds);
    else
        extractorFcn = @exampleBagOfFeaturesExtractor;
        bag = bagOfFeatures(imds, 'PointSelection', 'Detector', 'CustomExtractor',extractorFcn);
    end 
    
    save(fullfile(exportFolder,strcat('bag',Type,'.mat')),'bag');
end

% Test data
imds = imageDatastore(fullfile(testFolder, categories), 'LabelSource', 'foldernames');

% Train Data
fprintf("train Image Category Classifier....");
if exist(strcat(exportFolder,'\categoryClassifier',Type,'.mat'),'file') == 2
    load(fullfile(exportFolder,strcat('categoryClassifier',Type,'.mat')));
else
    categoryClassifier = trainImageCategoryClassifier(imds, bag);
    %save('minBow_imds.mat','imds');
    save(fullfile(exportFolder,strcat('categoryClassifier',Type,'.mat')),'categoryClassifier');
end

fprintf("Evaluate matrix....");
evaluate(categoryClassifier, imds);
%mean(diag(confMatrixTest))

end

