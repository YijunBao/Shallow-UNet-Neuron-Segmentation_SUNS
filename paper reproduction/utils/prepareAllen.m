%
% Please cite this paper if you use any component of this software:
% S. Soltanian-Zadeh, K. Sahingur, S. Blau, Y. Gong, and S. Farsiu, "Fast 
% and robust active neuron segmentation in two-photon calcium imaging using 
% spatio-temporal deep learning," Proceedings of the National Academy of Sciences (PNAS), 2019.
%
% Released under a GPL v2 license.

function [video] = prepareAllen(opt,DirData,DirSave)

% read data from h5 files, temporally bin and save files for the first
% 1/5th of the recordings
if exist(DirData)
    % read dimension of data
    fname = ['ophys_experiment_',opt.ID,'.h5'];
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
        %Crop border
        pixSize = 0.78; %um
        border = round(10/pixSize);
        video = video(border:end-border,border:end-border,:); 
        
        %Save cropped downsampled data
        if ~exist(DirSave)
            mkdir(DirSave);
        end
        
        h5create([DirSave,opt.ID,'.h5'],'/mov',size(video),'Datatype','uint16');
        h5write([DirSave,opt.ID,'.h5'],'/mov',video);
    end    
else
    error('Data Unavailable')
end

end

