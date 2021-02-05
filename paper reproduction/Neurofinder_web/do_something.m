%% Determine magnification according to neuron areas
list_list_Exp_ID={{'00.00', '00.01', '00.02', '00.03', '00.04', '00.05', ...
            '00.06', '00.07', '00.08', '00.09', '00.10', '00.11'}, ...
            {'01.00', '01.01'}, {'02.00', '02.01'}, ...
            {'03.00'}, {'04.00'}, {'05.00'}};

mean_areas = zeros(length(list_list_Exp_ID),1);
for lid = 1:length(list_list_Exp_ID)
    list_Exp_ID = list_list_Exp_ID{lid};
    list_areas = cell(length(list_Exp_ID),1);
    % folder of the raw video
    dir_video=['E:\NeuroFinder\web\train videos\',list_Exp_ID{1}(1:2),'\'];
    % folder of the GT Masks
    dir_GTMasks=fullfile(dir_video,'GT Masks');
    
    for vid=1:length(list_Exp_ID)
        Exp_ID = list_Exp_ID{vid};

        % Load ground truth masks
        load(fullfile(dir_GTMasks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
        list_areas{vid}=squeeze(sum(sum(FinalMasks)));
        clear FinalMasks;
    end
    mean_areas(lid) = mean(cell2mat(list_areas));
end

% %%
list_px_um = [1/1.15, 1/0.8, 1/1.15, 1.17, 0.8, 1.25]';
areas_compare = [mean_areas, 177*(0.785*list_px_um).^2, 177*(0.785./list_px_um).^2]; 


%% Read the output scores
list_list_Exp_ID={{'00.00', '00.01', '00.02', '00.03', '00.04', '00.05', ...
            '00.06', '00.07', '00.08', '00.09', '00.10', '00.11'}, ...
            {'01.00', '01.01'}, {'02.00', '02.01'}, ...
            {'03.00'}, {'04.00'}, {'05.00'}};

num_list = length(list_list_Exp_ID);
list_Tables = cell(num_list,1);
[F1_all, Recall_all, Precision_all, Time_frame_all,...
    minArea_all, avgArea_all, thresh_pmap_all, thresh_COM_all, ...
    thresh_IOU_all, thresh_consume_all, cons_all, win_avg_all] = deal(zeros(num_list,1));
Table_max = zeros(num_list,3);

for lid = 1:num_list % [0,4,5]+1 % 
    list_Exp_ID = list_list_Exp_ID{lid};
    % folder of the raw video
    dir_video=['E:\NeuroFinder\web\train videos\',list_Exp_ID{1}(1:2),'\'];
    % folder of the GT Masks
    dir_optim_info=fullfile(dir_video,'noSF\trial 2\output_masks');
    dir_output=fullfile(dir_video,'noSF\trial 2\output_masks online'); %  track no_update

    load(fullfile(dir_output,'Output_Info_All_offical.mat'));
    Table_time=[list_Recall, list_Precision, list_F1, list_time, list_time_frame];
    Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
    list_Tables{lid} = Table_time_ext;

    load(fullfile(dir_optim_info,['Optimization_Info_',num2str(length(list_Exp_ID)),'.mat']), 'Params','Table'); 
    minArea_all(lid)=Params.minArea;
    avgArea_all(lid)=Params.avgArea;
    thresh_pmap_all(lid)=Params.thresh_pmap;
    thresh_COM_all(lid)=Params.thresh_COM;
    thresh_IOU_all(lid)=Params.thresh_IOU;
    thresh_consume_all(lid)=Params.thresh_consume;
    cons_all(lid)=Params.cons;
    [~, indmax] = max(Table(:,end));
    Table_max(lid,:) = Table(indmax,end-2:end);
end
% list_Tables = cell2mat(list_Tables);
% for lid = [1,2,3]+1 % 1:num_list
%     list_Tables{lid} = zeros(3,9);
% end

list_Tables_summary = cell2mat(cellfun(@(x) x(end-1,1:3), list_Tables,'UniformOutput',false));
list_time_summary = cell2mat(cellfun(@(x) x(end-1,end), list_Tables,'UniformOutput',false));

Params_all=[minArea_all, avgArea_all, thresh_pmap_all, thresh_COM_all, ...
    cons_all, list_Tables_summary, Table_max, list_time_summary]; %, Time_frame_all
% Params_all = Params_all([1,5,6],:);
Params_all_ext=[Params_all;nanmean(Params_all,1);nanstd(Params_all,1,1)]; %([1,2,4:10],:)
disp(Params_all_ext(end-1,end-6:end-4))
