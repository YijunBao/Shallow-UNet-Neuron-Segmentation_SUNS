% Set the path of the 'GT Masks' folder, which contains the manual labels in 3D arrays.
dir_Masks = '..\demo\data\GT Masks\';

%%
dir_all = dir(fullfile(dir_Masks,'*FinalMasks*.mat'));
for ind = 1:length(dir_all)
    filename = dir_all(ind).name;
    if ~contains(filename,'_sparse')
        disp(filename)
        load(fullfile(dir_Masks,filename),'FinalMasks');
        [Lx,Ly,ncells]=size(FinalMasks);
        GTMasks_2=sparse(reshape(logical(FinalMasks),Lx*Ly,ncells));
        save(fullfile(dir_Masks,[filename(1:end-4),'_sparse.mat']),'GTMasks_2','-v7');
    end
end