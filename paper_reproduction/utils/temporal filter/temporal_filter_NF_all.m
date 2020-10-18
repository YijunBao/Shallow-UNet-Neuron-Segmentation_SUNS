%%
% name of the videos. Only use 04.01 and 04.01.test. 
tasks = {'train','test'};
list_Exp_ID={'01.00', '01.01', '02.00', '02.01', '04.00', '04.01'}; % '04.01';
list_rate_hz = [7.5, 7.5, 8, 8, 6.75, 3]; % frame rate
         
before=5; % number of frames before spike peak
after=30; % number of frames after spike peak
list_d=[3:10, 12:2:20, 25, 30, 100]; % [6,8]; % two element array showing the minimum and maximum allowed SNR
doesplot=true;
num_dff=length(list_d)-1;
[array_tau_s,array_tau2_s]=deal(length(tasks)*length(list_Exp_ID),num_dff);
spikes_avg_all=nan(length(tasks)*length(list_Exp_ID), before+after+1);
list_spikes_all=cell(length(tasks)*length(list_Exp_ID),num_dff);
time_frame = -before:after;

figure(97);
clf;
set(gcf,'Position',[100,100,500,400]);
hold on;

%%
for vid = 1:length(list_Exp_ID)
    for tid=1:length(tasks)
        id = vid+(tid-1)*length(list_Exp_ID);
        task = tasks{tid};
        % folder of the GT Masks
        dir_GTMasks=['E:\NeuroFinder\',task,' videos\'];
        %% Load traces and ROIs of all four sub-videos
        Exp_ID = list_Exp_ID{vid};
        if strcmp(task,'test')
            Exp_ID = [Exp_ID,'.test'];
        end
        fs = list_rate_hz(vid);
        load(['raw and bg traces ',Exp_ID,'.mat'],'traces_raw','traces_bg_exclude');
        traces_in = (traces_raw-traces_bg_exclude);
        load(fullfile(dir_GTMasks,['GTMasks_',Exp_ID,'_sparse.mat']),'GTMasks_2');
        ROIs2=logical(GTMasks_2);
        clear FinalMasks;

        %% Calculate the average spike shape and decay time
        for ii=1:num_dff
            [tau,tau2,spikes_avg,list_spikes_all{id,ii}]=determine_decay_time_d_nomix(traces_in, ...
                ROIs2, list_d(ii), list_d(ii+1), before, after, doesplot, fs);
            array_tau_s(id,ii)=tau/fs;
            array_tau2_s(id,ii)=tau2/fs;
        end
        spikes_avg_all(id,:)=spikes_avg;

        figure(97);
        plot(time_frame/fs, spikes_avg);
    end
end

%% plot the average spike shapes of each video with normalized amplitude
figure(97);
xlim([-1,4]);
legend(list_Exp_ID);

%% print the mean decay time
filter_tempolate=nanmean(spikes_avg_all);
tau_s_mean=mean(array_tau_s);
tau2_s_mean=mean(array_tau2_s);
tau_s_std=std(array_tau_s,1);
tau2_s_std=std(array_tau2_s,1);
fprintf('Decay time from e^{-1} peak: %f +- %f\n',tau_s_mean,tau_s_std);
fprintf('Decay time from 1/2 peak: %f +- %f\n',tau2_s_mean,tau2_s_std);

%% Save the filter template
% % save('GCaMP6s_spike_tempolate_mean.mat','filter_tempolate');
% h5create('GCaMP6s_spike_tempolate_mean.h5','/filter_tempolate',[1,1+before+after]);
% h5write('GCaMP6s_spike_tempolate_mean.h5','/filter_tempolate',filter_tempolate);

%% plot the average spike shapes of each video with normalized amplitude
figure;
t=(-before:after)/fs;
plot(t,spikes_avg_all','Color',[0.5,0.5,0.5]); 
hold on; 
plot(t,filter_tempolate,'r','LineWidth',2);
xlabel('Time (s)');
ylabel('Normalized Spike Amplitude');
title('Average Spike Tempolate');
plot(t,exp(-1)*ones(1,before+after+1),'b--');
% saveas(gcf,'GCaMP6s_spike_tempolate_mean.png');