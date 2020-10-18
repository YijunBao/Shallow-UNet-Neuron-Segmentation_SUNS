clear;
% folder of the raw video
dir_video='D:\ABO\20 percent\'; 
% folder of the GT Masks
dir_GTMasks='C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\';
% name of the videos
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};

for id = 1:length(list_Exp_ID)
    Exp_ID = list_Exp_ID{id};
    % Load video
    tic;
    fname=fullfile(dir_video,[Exp_ID,'.h5']);
    video_raw=h5read(fname, '/mov');
    toc;

    % Load ground truth masks
    load(fullfile(dir_GTMasks,['FinalMasks_FPremoved_',Exp_ID,'.mat']),'FinalMasks');
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
