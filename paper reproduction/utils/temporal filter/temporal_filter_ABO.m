% folder of the GT Masks
dir_GTMasks='C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\';
% name of the videos
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
         
fs=30; % frame rate
before=5; % number of frames before spike peak
after=30; % number of frames after spike peak
list_d=[6,8]; % two element array showing the minimum and maximum allowed SNR
doesplot=true;
num_dff=length(list_d)-1;
[array_tau,array_tau2]=deal(zeros(length(list_Exp_ID),num_dff));
spikes_avg_all=zeros(length(list_Exp_ID), before+after+1);
list_spikes_all=cell(10,num_dff);

%%
for id=1:length(list_Exp_ID)
    %% Load traces and ROIs of all four sub-videos
    Exp_ID = list_Exp_ID{id};
    load(['raw and bg traces ',Exp_ID,'.mat'],'traces_raw','traces_bg_exclude');
    traces_in = (traces_raw-traces_bg_exclude);
    load(fullfile(dir_GTMasks,['FinalMasks_FPremoved_',Exp_ID,'_sparse.mat']),'GTMasks_2');
    ROIs2=logical(GTMasks_2);
    clear FinalMasks;

    %% Calculate the average spike shape and decay time
    for ii=1:num_dff
        [tau,tau2,spikes_avg,list_spikes_all{id,ii}]=determine_decay_time_d_nomix...
            (traces_in, ROIs2, list_d(ii), list_d(ii+1), before, after, doesplot, fs);
        array_tau(id,ii)=tau;
        array_tau2(id,ii)=tau2;
    end
    array_tau_s=array_tau/fs;
    array_tau2_s=array_tau2/fs;
    spikes_avg_all(id,:)=spikes_avg;
end

%% print the mean decay time
filter_tempolate=mean(spikes_avg_all);
tau_s_mean=mean(array_tau_s);
tau2_s_mean=mean(array_tau2_s);
tau_s_std=std(array_tau_s,1);
tau2_s_std=std(array_tau2_s,1);
fprintf('Decay time from e^{-1} peak: %f +- %f\n',tau_s_mean,tau_s_std);
fprintf('Decay time from 1/2 peak: %f +- %f\n',tau2_s_mean,tau2_s_std);

%% Save the filter template
% save('GCaMP6f_spike_tempolate_mean.mat','filter_tempolate');
h5create('GCaMP6f_spike_tempolate_mean.h5','/filter_tempolate',[1,1+before+after]);
h5write('GCaMP6f_spike_tempolate_mean.h5','/filter_tempolate',filter_tempolate);

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
% saveas(gcf,'GCaMP6f_spike_tempolate_mean.png');

%% plot all single spikes with normalized amplitude
for ii=1:num_dff
    spikes_all = cell2mat(list_spikes_all(:,ii));
    figure;
    t=(-before:after)/fs;
    each = plot(t,spikes_all','Color',[0.5,0.5,0.5]); 
    hold on; 
    plot(t,filter_tempolate,'LineWidth',2);
    avg = plot(t,filter_tempolate,'LineWidth',2);
    xlabel('Time (s)'); %\DeltaF/F
    ylabel('Normalized Spike Amplitude');
    title('Spike Tempolate');
    exp_1 = plot([t(1),t(end)],exp(-1)*[1,1],'b--','LineWidth',1);
    zero = plot([-0.2,t(end)],0*[1,1],'k','LineWidth',1);
    
    set(gca,'FontSize',14,'LineWidth',1)
    ylim([-0.5,1]);
    legend([avg,each(1)],{'Average transient','individual transient'},'Location','South');
%     saveas(gcf,sprintf('GCaMP6f_spike_tempolate %d-%d.png',list_d(ii),list_d(ii+1)));
%     saveas(gcf,sprintf('GCaMP6f_spike_tempolate %d-%d.emf',list_d(ii),list_d(ii+1)));
end

