function bgtraces=generate_bgtraces_from_masks_exclude(video,masks)
% Generate background traces for each neuron from ground truth masks.
% Background intensity is the median of the intensity in the nearby 
% region of the neuron, but excluding the neuron itself.
[Lx,Ly,T]=size(video);
[Lxm,Lym,ncells]=size(masks);

[xx, yy] = meshgrid(1:Ly,1:Lx); 
r_cum=round(sqrt(mean(sum(sum(masks))))*1.5);

if Lx==Lxm && Ly==Lym
    video=reshape(video,[Lxm*Lym,T]);
else
    video=reshape(video(floor((Lx-Lxm)/2)+1:floor((Lx+Lxm)/2),floor((Ly-Lym)/2)+1:floor((Ly+Lym)/2),:),[Lxm*Lym,T]);
end

bgtraces=zeros(ncells,T,'single');
for nn=1:ncells
    mask=masks(:,:,nn);
    [xxs,yys]=find(mask>0);
    comx=mean(xxs);
    comy=mean(yys);
    circleout = (yy-comx).^2 + (xx-comy).^2 < r_cum^2;
    circleout_exclude = circleout & ~mask;
    bgtraces(nn,:)=median(video(circleout_exclude(:),:),1);
end


