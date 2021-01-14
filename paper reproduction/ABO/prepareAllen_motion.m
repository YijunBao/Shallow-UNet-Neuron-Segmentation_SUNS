%
% Please cite this paper if you use any component of this software:
% S. Soltanian-Zadeh, K. Sahingur, S. Blau, Y. Gong, and S. Farsiu, "Fast 
% and robust active neuron segmentation in two-photon calcium imaging using 
% spatio-temporal deep learning," Proceedings of the National Academy of Sciences (PNAS), 2019.
%
% Released under a GPL v2 license.

function [video] = prepareAllen_motion(opt,DirData,DirSave,sigma,fs,f_lpf)

% read data from h5 files, temporally bin and save files for the first
% 1/5th of the recordings
if exist(DirData)
    % read dimension of data
    fname = ['ophys_experiment_',opt.ID,'.h5'];%'_concat_31Hz_0.h5'];
    infoVid = h5info([DirData,fname]);
    Nx = infoVid.Datasets.Dataspace.Size(1); 
    Ny = infoVid.Datasets.Dataspace.Size(2);
    Nframes = infoVid.Datasets.Dataspace.Size(3);
    
    % Process the entire data in 5 intervals
    startT = [1,floor(Nframes/5)*[1:4]+1];
    
    for ii = 1:1
        startS = [1,1,startT(ii)];
        video = h5read([DirData,fname],...
            '/data',startS,[Nx,Ny,floor(Nframes/5)],[1,1,1]); 
        %Add random shift
        T = floor(Nframes/5);
        cut_lpf = round(f_lpf/fs*T);
        max_shift = 5*sigma; % maximum shift in each direction
        lpf = zeros(T,1);
        lpf([1:1+cut_lpf,end-cut_lpf+1:end])=1;
        shift = randn(T,2)*sigma;
        shift_lpf = real(ifft(fft(shift).*lpf));
        shift_lpf(shift_lpf>max_shift)=max_shift;
        shift_lpf(shift_lpf<-max_shift)=-max_shift;
        for t = 1:T
            video(:,:,t) = imtranslate(video(:,:,t), shift_lpf(t,:));
        end
        
        %Crop border
        pixSize = 0.78; %um
        border = round(10/pixSize);
        video = video(border:end-border,border:end-border,:); 
        
        %Temporal binarization by summation
        video = uint16(binVideo_temporal(video,opt.ds));
        
        %Save cropped downsampled data
        if ~exist(DirSave)
            mkdir(DirSave);
        end
        
%         niftiwrite(video,[DirSave,opt.ID,'_processed'],'compressed',false);
        h5create([DirSave,opt.ID,'.h5'],'/mov',size(video),'Datatype','uint16');
        h5write([DirSave,opt.ID,'.h5'],'/mov',video);
    end    
else
    error('Data Unavailable')
end

end

