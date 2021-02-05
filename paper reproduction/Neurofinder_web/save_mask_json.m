addpath(genpath('C:\Matlab Files\jsonlab'));

%% Save each .mat output to .json files
% Set the path of the 'GT Masks' folder, which contains the manual labels in 3D arrays.
% dir_data = 'E:\NeuroFinder\web\train videos\*\GT Masks\';
dir_data = 'E:\NeuroFinder\web\train videos\*\noSF\trial*\output_masks*\';

dir_all = dir(fullfile(dir_data,'*Output_Masks*.mat'));
for ind = 1:length(dir_all)
    filename = dir_all(ind).name;
    if ~contains(filename,'_sparse')
        disp(filename)
        load(fullfile(dir_all(ind).folder,filename),'Masks'); % 'FinalMasks'
        FinalMasks = permute(Masks,[3,2,1]);
        [Lx,Ly,ncells]=size(FinalMasks);
        mat2json = cell(ncells,1);
        for m = 1:ncells
            mask = FinalMasks(:,:,m);
            [x,y] = find(mask);
            coordinates = [x,y]-1;
            mat2json{m} = struct("coordinates", coordinates);
        end
        mat2json = [mat2json{:}];
        savejson('',mat2json,fullfile(dir_all(ind).folder,[filename(1:end-4),'.json']));
%         GTMasks_2=sparse(reshape(logical(FinalMasks),Lx*Ly,ncells));
%         save(fullfile(dir_all(ind).folder,[filename(1:end-4),'_sparse.mat']),'GTMasks_2');
    end
end

%% Save all output files to a single .json file
dir_data = 'E:\NeuroFinder\web\test videos\';
list_Exp_ID = {'00.00', '00.01', '01.00', '01.01', '02.00', '02.01', '03.00', '04.00', '05.00'};
num_video = length(list_Exp_ID);
results = cell(num_video,1);
for ind = 1:num_video
    Exp_ID = list_Exp_ID{ind};
    if ind==num_video
        dataset = '04.01.test';
    else
        dataset = [Exp_ID,'.test'];
    end
    load([dir_data,Exp_ID(1:2),'\noSF\output_masks online\Output_Masks_',Exp_ID,'.mat'],'Masks');
    FinalMasks = permute(Masks,[3,2,1]);
    [Lx,Ly,ncells]=size(FinalMasks);
    regions = cell(ncells,1);
    for m = 1:ncells
        mask = FinalMasks(:,:,m);
        [x,y] = find(mask);
        coordinates = [x,y]-1;
        regions{m} = struct("coordinates", coordinates);
    end
    results{ind} = struct('dataset',dataset, 'regions',[regions{:}]);
end
results = [results{:}];
savejson('', results, 'results.json');
