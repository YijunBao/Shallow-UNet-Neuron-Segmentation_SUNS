clear;
tasks = {'train'};
for tid = 1:length(tasks)
    task = tasks{tid};
    % folder of the raw video
    dir_video=['E:\NeuroFinder\web\',task,' videos\'];
    % folder of the GT Masks
    dir_GTMasks=fullfile(dir_video,'GT Masks');
    % name of the videos
    list_Exp_ID={'01.00', '01.01'};

    for vid=1:length(list_Exp_ID)
        Exp_ID = list_Exp_ID{vid};
        if strcmp(task,'test')
            Exp_ID = [Exp_ID,'.test'];
        end
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

        dir_trace = fullfile(dir_video,'traces');
        if ~exist(dir_trace)
            mkdir(dir_trace)
        end
        save(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude');
    end
end