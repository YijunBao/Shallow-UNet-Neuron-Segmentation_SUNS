clear;
% folder of the raw video
dir_video='F:\CaImAn data\WEBSITE\divided_data\';
% name of the videos
list_Exp_ID={'J115', 'J123', 'K53', 'YST'};

for vid= 1:length(list_Exp_ID)
    Exp_ID = list_Exp_ID{vid};
    for xpart = 1:2
        for ypart = 1:2
            Exp_ID_part = sprintf('%s_part%d%d',Exp_ID,xpart,ypart);
            % Load video
            tic;
            fname=fullfile(dir_video,Exp_ID,[Exp_ID_part,'.h5']);
            video_raw=h5read(fname, '/mov');
            toc;

            % Load ground truth masks
            load(fullfile(dir_video,Exp_ID,['FinalMasks_',Exp_ID_part,'.mat']),'FinalMasks');
            ROIs=logical(FinalMasks);
            clear FinalMasks;
            
           %%
            tic; 
            traces_raw=generate_traces_from_masks(video_raw,ROIs);
            toc;
            traces_bg_exclude=generate_bgtraces_from_masks_exclude(video_raw,ROIs);
            toc;

            save(['raw and bg traces ',Exp_ID_part,'.mat'],'traces_raw','traces_bg_exclude');
        end
    end
end
