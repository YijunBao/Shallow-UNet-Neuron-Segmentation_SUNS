function traces=generate_traces_from_masks(video,masks)
% Generate traces for each neuron from ground truth masks
[Lx,Ly,T]=size(video);
[Lxm,Lym,ncells]=size(masks);

if Lx==Lxm && Ly==Lym
    video=reshape(video,[Lxm*Lym,T]);
else
    video=reshape(video(floor((Lx-Lxm)/2)+1:floor((Lx+Lxm)/2),floor((Ly-Lym)/2)+1:floor((Ly+Lym)/2),:),[Lxm*Lym,T]);
end
masks=reshape(masks,[Lxm*Lym,ncells]);
traces = single(zeros(ncells, T));
for k = 1:ncells
    traces(k, :) = mean(video(masks(:,k)>0, :), 1);
end

