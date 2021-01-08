% Set the path of the 'GT Masks' folder, which contains the manual labels in 3D arrays.
% dir_data = 'C:\Users\baoyi\Documents\GitHub\Shallow-UNet-Neuron-Segmentation_SUNS\demo\data\GT Masks\';
dir_data = 'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\Grader4\';

%%
dir_all = dir(fullfile(dir_data,'*FinalMasks*.mat'));
for ind = 1:length(dir_all)
    filename = dir_all(ind).name;
    if ~contains(filename,'_sparse')
        disp(filename)
        load(fullfile(dir_data,filename),'FinalMasks');
        [Lx,Ly,ncells]=size(FinalMasks);
        GTMasks_2=sparse(reshape(logical(FinalMasks),Lx*Ly,ncells));
        save(fullfile(dir_data,[filename(1:end-4),'_sparse.mat']),'GTMasks_2');
    end
end