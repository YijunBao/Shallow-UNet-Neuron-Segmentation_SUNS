function [tau1,tau2,spikes_avg,list_spikes_all]=determine_decay_time_d_nomix...
    (traces_raw,ROIs2, d_min, d_max, before, after,doesplot, fs)
% Function to calculate decay time and average spike shape from a moderate SNR range.
% Inputs: 
%     traces_raw: traces of each neuron. length = (ncells, T)
%     ROIs2: neuron masks in 2D format. length = (Lx*Ly, ncells)
%     d_min: minimum peak SNR value
%     d_max: maximum peak SNR value
%     before: number of frames before spike peak
%     after: number of frames after spike peak
%     doesplot: true to plot the spikes
%     fs: frame rate of the video
% Outputs:
%     tau1: Decay time from e^{-1} peak
%     tau2: Decay time from 1/2 peak
%     spikes_avg: Average spike
%     list_spikes_all: All the selected spikes

[ncells,T]=size(traces_raw);
fine_ratio=10;
if ncells>T
    traces_raw=traces_raw';
    [ncells,T]=size(traces_raw);
end

list_spikes=cell(ncells,1);
F0=median(traces_raw,2);
% if min(F0)<=0
%     warning('Please input raw trace rather than df/f or SNR');
% end
sigma = median(abs(traces_raw-F0),2)/(sqrt(2)*erfinv(1/2)); 
d=(traces_raw-F0)./sigma; % SNR trace

onlyone=d>d_min;
% Remove the spikes that are coactivate with its neighbours. 
for tt=1:T
    active=find(onlyone(:,tt));
    sum_masks=sum(ROIs2(:,active),2);
    overlap=sum_masks>1;
    if any(overlap,'all')
        for aa=active
            if any(ROIs2(:,aa) & overlap,'all')
                onlyone(aa,tt)=0;
            end
        end
    end
end

% Select the spikes whose peak SNR is between d_min and d_max
good_amp=(d>d_min & d<d_max);
good_amp_diff=diff(padarray(good_amp,[0,1],false,'both'),1,2);

for nn=1:ncells
    good_start=find(good_amp_diff(nn,:)==1);
    good_end=find(good_amp_diff(nn,:)==-1)-1;
    if isempty(good_start)
        continue;
    end
    num_good=length(good_start);
    low_between=false(1,num_good+1);
    sep=1:good_start(1)-1;
    % low_between is used to ensure sufficient temporal distance between two spikes
    low_between(1)=~sum(d(nn,sep)>d_max);
    for pp=2:num_good
        sep=good_end(pp-1)+1:good_start(pp)-1;
        low_between(pp)=~sum(d(nn,sep)>d_max);
    end
    sep=good_end(num_good)+1:T;
    low_between(num_good+1)=~sum(d(nn,sep)>d_max);
    good_amp_dist=[good_start(1)-1,good_start(2:end)-good_end(1:end-1),T-good_end(end)];
    % good_amp_dist=circshift(good_amp_start,[-1,0])-good_amp_end;
    good_amp_dist=good_amp_dist.*low_between;
    good_dist=(good_amp_dist(1:end-1)>=before+after) & (good_amp_dist(2:end)>=before+after);
    good_ind=find(good_dist);
    num_good=length(good_ind);
    
    spikes=zeros(num_good,1+before+after);
    for jj=1:num_good
        pp=0;
        onlyone_current=onlyone(nn,good_start(good_ind(jj)):good_end(good_ind(jj)));
        if all(onlyone_current)
            spike_current=d(nn,good_start(good_ind(jj))-1:good_end(good_ind(jj))+1);
            [peaks, locations]=findpeaks(spike_current);
            if ~isempty(peaks) && length(peaks)==1
                pp=pp+1;
                location_all=good_start(good_ind(jj))-2+locations;
                spikes(pp,:)=d(nn,location_all-before:location_all+after)/peaks; %
            end
            list_spikes{nn}=spikes(1:pp,:);
        end
    end
end

list_spikes_all=cell2mat(list_spikes);
if isempty(list_spikes_all)
    tau1 = nan;
    tau2 = nan;
    spikes_avg = nan(1+before+after,1);
else
    spikes_avg=mean(list_spikes_all,1);
    t_fine=(1:(before+after+1)*fine_ratio)/fine_ratio;
    spikes_fine=interp1(1:(before+after+1),spikes_avg,t_fine); %,'pchip'
    taud1=find(spikes_fine>exp(-1),1,'last');
    tau1=taud1/fine_ratio-before-1;
    % spikes_fine_subs=spikes_fine-spikes_fine(end);
    taud2=find(spikes_fine>0.5,1,'last');
    tau2=taud2/fine_ratio-before-1;
    tau2=tau2/log(2);
    num_avg=size(list_spikes_all,1);

    time=(-before:after)/fs;
    if doesplot % plot all the selected spikes with normalized amplitude
        figure; 
        plot(time,list_spikes_all','Color',[0.5,0.5,0.5]);
        hold on;
        plot((t_fine-before-1)/fs,spikes_fine,'LineWidth',2);
        title(['Average trace from ',num2str(num_avg),' spikes, ',num2str(d_min),'<d<',num2str(d_max)]);
        xlabel('Time (s)');
        ylabel('Normalized d');
    end
end
