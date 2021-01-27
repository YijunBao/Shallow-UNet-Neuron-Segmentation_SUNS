%
% Please cite this paper if you use any component of this software:
% S. Soltanian-Zadeh, K. Sahingur, S. Blau, Y. Gong, and S. Farsiu, "Fast 
% and robust active neuron segmentation in two-photon calcium imaging using 
% spatio-temporal deep learning," Proceedings of the National Academy of Sciences (PNAS), 2019.
%
% Released under a GPL v2 license.

function imgs = prepareNeurofinder(opt,DirData,DirSave)

ID = opt.ID;
       
if strcmp(opt.type,'test')
    name = ['neurofinder.',ID,'.',opt.type];
else
    name = ['neurofinder.',ID];
end
    
if exist(fullfile(DirData,name))

    files = dir(fullfile(DirData,name,'images','*.tiff'));
    fname = strcat(fullfile(DirData,name,'images',files(1).name));
    img1 = imread(fname);
    [Lx, Ly] = size(img1);
    imgs = zeros(Lx, Ly, length(files),'uint16');

    for i = 1:length(files)
        fname = strcat(fullfile(DirData,name,'images',files(i).name));
        imgs(:,:,i) = imread(fname);
    end

    %save cropped and downsampled data
    if ~exist(DirSave)
        mkdir(DirSave)
    end

    h5create(fullfile(DirSave,[ID,'.h5']),'/mov',size(imgs),'Datatype','uint16');
    h5write(fullfile(DirSave,[ID,'.h5']),'/mov',imgs);
else
    error('Data not available.');
end

end

