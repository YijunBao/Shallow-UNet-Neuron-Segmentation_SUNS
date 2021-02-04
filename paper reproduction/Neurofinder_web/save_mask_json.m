% Set the path of the 'GT Masks' folder, which contains the manual labels in 3D arrays.
% dir_data = 'E:\NeuroFinder\web\train videos\*\GT Masks\';
dir_data = 'E:\NeuroFinder\web\train videos\*\complete\output_masks\';

%%
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
