%%
list_rate_hz = [7, 7.5, 8, 7.5, 6.75, 3];
% name of the videos. Only use 04.01 and 04.01.test. 
list_Exp_ID = {'01.00','01.01'};
rate_hz = list_rate_hz(2); % frame rate
         
before=15; % number of frames before spike peak
after=40; % number of frames after spike peak
list_d=[6,8]; % two element array showing the minimum and maximum allowed SNR
doesplot=true;
num_dff=length(list_d)-1;
[array_tau_s,array_tau2_s]=deal(2,num_dff);
spikes_avg_all=nan(2, before+after+1);
time_frame = -before:after;

figure(97);
clf;
set(gcf,'Position',[100,100,500,400]);
hold on;

%%
tasks = 'train'; % {'train','test'};
dir_video=['E:\NeuroFinder\web\',task,' videos\'];
dir_trace = fullfile(dir_video,'traces');
dir_GTMasks = fullfile(dir_video,'GT Masks');

for vid=1:length(list_Exp_ID)
    % folder of the GT Masks
    %% Load traces and ROIs of all four sub-videos
    Exp_ID = list_Exp_ID{vid};
    if strcmp(task,'test')
        Exp_ID = [Exp_ID,'.test'];
    end
    fs = rate_hz;
    load(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude');
    traces_in = (traces_raw-traces_bg_exclude);
    load(fullfile(dir_GTMasks,['FinalMasks_',Exp_ID,'_sparse.mat']),'GTMasks_2');
    ROIs2=logical(GTMasks_2);
    clear FinalMasks;

    %% Calculate the average spike shape and decay time
    for ii=1:num_dff
        [tau,tau2,spikes_avg]=determine_decay_time_d_nomix(traces_in, ROIs2, list_d(ii), list_d(ii+1), before, after, doesplot, fs);
        array_tau_s(vid,ii)=tau/fs;
        array_tau2_s(vid,ii)=tau2/fs;
    end
    spikes_avg_all(vid,:)=spikes_avg;
    
    figure(97);
    plot(time_frame/fs, spikes_avg);
end

%% plot the average spike shapes of each video with normalized amplitude
figure(97);
% xlim([-1,4]);
legend(list_Exp_ID);

%% print the mean decay time
filter_tempolate=mean(spikes_avg_all);
tau_s_mean=mean(array_tau_s);
tau2_s_mean=mean(array_tau2_s);
tau_s_std=std(array_tau_s,1);
tau2_s_std=std(array_tau2_s,1);
fprintf('Decay time from e^{-1} peak: %f +- %f\n',tau_s_mean,tau_s_std);
fprintf('Decay time from 1/2 peak: %f +- %f\n',tau2_s_mean,tau2_s_std);

%% Save the filter template
saved_h5 = [Exp_ID(1:2),'_spike_tempolate.h5'];
% save('GCaMP6s_spike_tempolate_mean.mat','filter_tempolate');
if exist(saved_h5, 'file')
    delete(saved_h5)
end
h5create(saved_h5,'/filter_tempolate',[1,1+before+after]);
h5write(saved_h5,'/filter_tempolate',filter_tempolate);

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