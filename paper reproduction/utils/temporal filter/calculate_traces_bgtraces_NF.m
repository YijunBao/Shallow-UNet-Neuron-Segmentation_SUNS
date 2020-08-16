clear;
tasks = {'train','test'};
for tid = 1:2
    task = tasks{tid};
    % folder of the raw video
    dir_video=['E:\NeuroFinder\',task,' videos\'];
    % folder of the GT Masks
    dir_GTMasks=['C:\Matlab Files\STNeuroNet-master\Markings\Neurofinder\',task,'\Grader1\'];
    % name of the videos
    list_Exp_ID={'04.01'}; % '01.00', '01.01', '02.00', '02.01', '04.00', 

    for vid=1:length(list_Exp_ID)
        Exp_ID = list_Exp_ID{vid};
        if strcmp(task,'test')
            Exp_ID = [Exp_ID,'.test'];
        end
        % Load video
        tic;
        fname=[dir_video,Exp_ID,'.h5'];
        video_raw=h5read(fname, '/mov');
        toc;

        % Load ground truth masks
        load(fullfile(dir_GTMasks,['FinalMasks_',Exp_ID([2,4,5]),'.mat']),'FinalMasks');
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
end