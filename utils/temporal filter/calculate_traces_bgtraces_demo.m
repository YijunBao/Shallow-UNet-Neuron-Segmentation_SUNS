clear;
% folder of the raw video
dir_video='..\..\demo\data\'; 
% folder of the GT Masks
dir_GTMasks=fullfile(dir_video,'GT Masks\');
% name of the videos
list_Exp_ID={'YST_part11';'YST_part12';'YST_part21';'YST_part22'};

for id = 1:length(list_Exp_ID)
    Exp_ID = list_Exp_ID{id};
    % Load video
    tic;
    fname=fullfile(dir_video,[Exp_ID,'.h5']);
    video_raw=h5read(fname, '/mov');
    toc;

    % Load ground truth masks
    load(fullfile(dir_GTMasks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    ROIs=logical(FinalMasks);
    clear FinalMasks;

    %%
    tic; 
    traces_raw=generate_traces_from_masks(video_raw,ROIs);
    toc;
    traces_bg_exclude=generate_bgtraces_from_masks_exclude(video_raw,ROIs);
    toc;
    
    save(['raw and bg traces ',Exp_ID,'.mat'],'traces_raw','traces_bg_exclude');
end
