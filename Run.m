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
%%HOG_Features(rootFolder, testFolder, exportFolder);

%% Run Deep_Learning
%%DeepLearning(rootFolder, testFolder,exportFolder);

%% Bag Of Word
%Type = 'BoW';       % 1.all parameter is default value
Type = 'CustomBow'; % 2. 'Detector' VocabularySize = 1000
BagOfWord(rootFolder, testFolder, exportFolder, Type);