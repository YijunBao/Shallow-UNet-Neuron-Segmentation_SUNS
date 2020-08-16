% folder of the GT Masks
dir_GTMasks = 'F:\CaImAn data\WEBSITE\divided_data\';
% name of the videos
list_Exp_ID={'J115', 'J123', 'K53', 'YST'}; 
rate_hz = [30, 30, 30, 10]; % frame rate of each video
         
before=15; % number of frames before spike peak
after=60; % number of frames after spike peak
list_d=[5,6]; % two element array showing the minimum and maximum allowed SNR
doesplot=true;
num_dff=length(list_d)-1;
[array_tau,array_tau2]=deal(nan(length(list_Exp_ID),num_dff));
spikes_avg_all=nan(length(list_Exp_ID), before+after+1);
time_frame = -before:after;

figure(97);
clf;
set(gcf,'Position',[100,100,500,400]);
hold on;

%%
for vid=1:length(list_Exp_ID)
    %% Load traces and ROIs of all four sub-videos
    Exp_ID = list_Exp_ID{vid};
    fs = rate_hz(vid);
    traces_in = cell(4,1);
    ROIs = cell(1,1,4);
    part = 0;
    for xpart = 1:2
        for ypart = 1:2
            part = part+1;
            Exp_ID_part = sprintf('%s_part%d%d',Exp_ID,xpart,ypart);
            load(['raw and bg traces ',Exp_ID_part,'.mat'],'traces_raw','traces_bg','traces_bg_exclude');
            traces_in{part,1} = (traces_raw-traces_bg_exclude);
            load(fullfile(dir_GTMasks,Exp_ID,['FinalMasks_',Exp_ID_part,'.mat']),'FinalMasks');
            ROIs{1,1,part}=logical(FinalMasks);
            clear FinalMasks;
        end
    end
    traces_in = cell2mat(traces_in);
    ROIs = cell2mat(ROIs);
    [Lx,Ly,N]=size(ROIs);
    ROIs2 = reshape(ROIs,Lx*Ly,N);

    %% Calculate the average spike shape and decay time
    for ii=1:num_dff
        [tau,tau2,spikes_avg]=determine_decay_time_d_nomix(traces_in, ...
            ROIs2, list_d(ii), list_d(ii+1), before, after, doesplot, fs);
        array_tau(vid,ii)=tau;
        array_tau2(vid,ii)=tau2;
    end
    array_tau_s=array_tau/fs;
    array_tau2_s=array_tau2/fs;
    spikes_avg_all(vid,:)=spikes_avg;
    
    figure(97);
    plot(time_frame/fs, spikes_avg_all(vid,:), 'LineWidth',2);
end

%% plot the average spike shapes of each video with normalized amplitude
figure(97);
legend(list_Exp_ID);
title('Average Spike Shapes');
xlabel('Time(s)')
% saveas(gcf,'Average Spike Shapes CM.png');
% save('CaImAn spkie template.mat','spikes_avg_all');

%% bar plot of the decay time of each video
figure;
bar(array_tau_s);
xlabel('Video id');
ylabel('Decay time, \tau (s)');
title(['CaImAn videos, Decay time from e^{-1} peak']);
xticklabels(list_Exp_ID);
% saveas(gcf,['CaImAn tau e-1 peak.png']);

figure;
bar(array_tau2_s);
xlabel('Video id');
ylabel('Decay time, \tau (s)');
title(['CaImAn videos, Decay time from 1/2 peak']);
xticklabels(list_Exp_ID);
% saveas(gcf,['CaImAn tau 2-1 peak.png']);

%% Save the filter template
for vid=1:length(list_Exp_ID)
    Exp_ID = list_Exp_ID{vid};
    filter_tempolate = spikes_avg_all(vid,:);
%     save([Exp_ID,'_spike_tempolate.mat'],'filter_tempolate');
    h5_name = [Exp_ID,'_spike_tempolate.h5'];
    if exist(h5_name, 'file')
        delete(h5_name)
    end
    h5create([Exp_ID,'_spike_tempolate.h5'],'/filter_tempolate',[1,1+before+after]);
    h5write([Exp_ID,'_spike_tempolate.h5'],'/filter_tempolate',filter_tempolate);
end

