%% track on raw video
list_Exp_ID={'YST_part11';'YST_part12';'YST_part21';'YST_part22'};

k=4;
dir_video = 'data\';
dir_mask = [dir_video,'noSF\'];
nframes = 300;
fps = 10;
merge = 10;
start=[1,1,300];
count=[Inf,Inf,nframes];
stride=[1,1,1];

color_range_raw = [500,2000];
Lx=88; Ly=120;
% Lxc=210; Lyc=120;
% rangex=200:(200+210-1); rangey=30:(30+120-1);
rangex=1:Ly; rangey=1:Lx;
crop_png_1=[155,55,100,150];
crop_png_2=[255,55,45,150];
crop_png_3=[300,55,15,150];
% crop_img=[86,64,Lyc,Lxc];

Exp_ID = list_Exp_ID{k};
video_raw = h5read([dir_video,Exp_ID,'.h5'],'/mov',start, count, stride);
load([dir_mask,'output_masks online video\Output_Masks_',Exp_ID,'.mat'],'list_Masks_2');
load([dir_mask,'output_masks track video\Output_Masks_',Exp_ID,'.mat'],'list_active_old','list_Masks_cons_2D');
list_active_old = list_active_old';
list_Masks_cons_2D = list_Masks_cons_2D';
%%
mask = sum(reshape(full(list_Masks_2{1}'),Lx,Ly,[]),3);
mask_track = sum(reshape(full(list_Masks_cons_2D{1}'),Lx,Ly,[]),3);
% border = 255*ones(Lxc,10,3,'uint8');
v = VideoWriter('Masks 2 raw.avi');
v.FrameRate = fps;
open(v);
figure('Position',[100,100,380,260],'Color','w');

for t = 1:nframes
    image = video_raw(:,:,t);
    
    clf; % masks from SUNS online
    imshow(image(rangey,rangex)', color_range_raw);
    if mod(t-4,fps)==0
        t_mask = floor((t-4)/fps)+1;
        mask = sum(reshape(full(list_Masks_2{t_mask}'),Lx,Ly,[]),3);
    end
    set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
    h=colorbar;
    set(h,'FontSize',9);
    set(get(h,'Label'),'String','Raw intensity','FontName','Arial');
    title('No tracking')
    hold on;
    contour(mask(rangey,rangex)','Color', [0.9,0.1,0.1]);
    pause(0.001);
    img_all=getframe(gcf,crop_png_1);
    img_notrack=img_all.cdata;
    
    clf; % masks from tracking SUNS online
    imshow(image(rangey,rangex)', color_range_raw);
    mask_track = reshape(full(list_Masks_cons_2D{t}'),Lx,Ly,[]);
    set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
    h=colorbar;
    set(get(h,'Label'),'String','Raw intensity','FontName','Arial');
    set(h,'FontSize',9);
    title('Tracking')
    hold on;
    contour(sum(mask_track(rangey,rangex,:),3)','Color', [0.9,0.1,0.1]);
    contour(sum(mask_track(rangey,rangex,list_active_old{t}'),3)','Color', [0.1,0.9,0.1]);
    pause(0.001);
    img_all=getframe(gcf,crop_png_1);
    img_track=img_all.cdata;
    if t==1
        img_all=getframe(gcf,crop_png_2);
        img_colorbar=img_all.cdata;
        img_all=getframe(gcf,crop_png_3);
        img_label=img_all.cdata;
    end
    
    img_both = cat(2,img_notrack, img_track, img_colorbar, img_label);
%     figure(99); imshow(img_both);
    writeVideo(v,img_both);
end
close(v);

%% track on SNR video
% list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
%              '510214538';'524691284';'527048992';'531006860';'539670003'};

% k=1;
% dir_video = 'D:\ABO\20 percent\ShallowUNet\complete\';
% dir_mask = [dir_video,'std7_nf200_ne200_bs20\DL+100FL(1,0.25)\'];
% nframes = 450;
% merge = 30;
% start=[1,1,900];
% count=[Inf,Inf,nframes];
% stride=[1,1,1];

color_range_SNR = [0,5];
% Lx=487; Ly=487;
crop_png_1=[155,55,100,150];
crop_png_2=[255,55,60,150];
% crop_png_3=[300,55,15,150];


Exp_ID = list_Exp_ID{k};
video_SNR = h5read([dir_video,'noSF\network_input\',Exp_ID,'.h5'],'/network_input',start, count, stride);
% load([dir_mask,'output_masks online video\Output_Masks_',Exp_ID,'.mat'],'list_Masks_2');
% load([dir_mask,'output_masks track video\Output_Masks_',Exp_ID,'.mat'],'list_active_old');
% list_active_old = list_active_old';
% load([dir_mask,'output_masks track video\Output_Masks_',Exp_ID,'_0.mat'],'list_Masks_cons_2D_0');
% list_Masks_cons_2D_0 = list_Masks_cons_2D_0';
% %%
mask = sum(reshape(full(list_Masks_cons_2D{1}'),Lx,Ly,[]),3);
mask_track = sum(reshape(full(list_Masks_cons_2D{1}'),Lx,Ly,[]),3);
v = VideoWriter('Masks 2 SNR.avi');
v.FrameRate = fps;
open(v);
figure('Position',[100,100,380,260],'Color','w');

for t = 1:nframes
    image = video_SNR(:,:,t);
    
    clf; % masks from SUNS online
    imshow(image(rangey,rangex)', color_range_SNR);
    if mod(t-4,fps)==0
        t_mask = floor((t-4)/fps)+1;
        mask = sum(reshape(full(list_Masks_2{t_mask}'),Lx,Ly,[]),3);
    end
    set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
    h=colorbar;
    set(h,'FontSize',9);
    set(get(h,'Label'),'String','SNR','FontName','Arial');
    title('No tracking')
    hold on;
    contour(mask(rangey,rangex)','Color', [0.9,0.1,0.1]);
    pause(0.001);
    img_all=getframe(gcf,crop_png_1);
    img_notrack=img_all.cdata;
    
    clf; % masks from tracking SUNS online
    imshow(image(rangey,rangex)', color_range_SNR);
    mask_track = reshape(full(list_Masks_cons_2D{t}'),Lx,Ly,[]);
    set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
    h=colorbar;
    set(get(h,'Label'),'String','SNR','FontName','Arial');
    set(h,'FontSize',9);
    title('Tracking')
    hold on;
    contour(sum(mask_track(rangey,rangex,:),3)','Color', [0.9,0.1,0.1]);
    contour(sum(mask_track(rangey,rangex,list_active_old{t}'),3)','Color', [0.1,0.9,0.1]);
    pause(0.001);
    img_all=getframe(gcf,crop_png_1);
    img_track=img_all.cdata;
    if t==1
        img_all=getframe(gcf,crop_png_2);
        img_colorbar=img_all.cdata;
%         img_all=getframe(gcf,crop_png_3);
%         img_label=img_all.cdata;
    end
    
    img_both = cat(2,img_notrack, img_track, img_colorbar); %, img_label
%     figure(99); imshow(img_both);
    writeVideo(v,img_both);
end
close(v);

    