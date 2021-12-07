% Set the path of the 'GT Masks' folder, which contains the manual labels in 3D arrays.
dir_in = 'D:\ABO\20 percent 200\complete\output_masks\';
dir_out = 'D:\ABO\20 percent 200\SUNS_complete Masks\';
if ~exist(dir_out,'dir')
    mkdir(dir_out)
end

%%
dir_all = dir(fullfile(dir_in,'*Output_Masks*.mat'));
for ind = 1:length(dir_all)
    filename = dir_all(ind).name;
    disp(filename)
    load(fullfile(dir_in,filename),'Masks');
    [Lx,Ly,ncells]=size(Masks);
    FinalMasks=permute(Masks,[3,2,1]);
    save(fullfile(dir_out,['Final',filename(8:end-4),'.mat']),'FinalMasks','-v7');
    % Please do not save them in 'v7.3' format, otherwise, 
    % you may need to use h5sparese to read the files in python.
%     end
end