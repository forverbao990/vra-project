%% Constant
rootFolder = 'cifar10Train';
testFolder = 'cifar10Test';
exportFolder = 'export';

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

%% Run HOG_Features file
% Extract HOG features and HOG visualization

%cellSize = [4 4];
%ellType = '4x4';

%cellSize = [8 8];
%cellType = '8x8';
%HOG_Features(rootFolder, testFolder, exportFolder, cellSize, cellType);

%% Run Deep_Learning

%featureLayer = 'fc7'; %4096 fully connected layer
%featureLayer = 'fc8'; % 1000 fully connected layer
%featureLayer = 'conv4'; %Convolution: 384 3x3x192 convolutions with stride [1  1] and padding [1  1]
%DeepLearning(rootFolder, testFolder,exportFolder, featureLayer);

%% Bag Of Word
%Type = 'BoW';       % 1.all parameter is default value
%Type = 'CustomBow'; % 2. 'Detector' VocabularySize = 1000
%BagOfWord(rootFolder, testFolder, exportFolder, Type);