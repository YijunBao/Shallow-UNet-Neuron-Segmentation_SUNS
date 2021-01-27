% example MATLAB script for loading neurofinder data
%
% for more info see:
%
% - http://neurofinder.codeneuro.org
% - https://github.com/codeneuro/neurofinder
%
% requires one package from the matlab file exchange
%
% - jsonlab
% - http://www.mathworks.com/matlabcentral/fileexchange/33381-jsonlab--a-toolbox-to-encode-decode-json-files-in-matlab-octave
%%
addpath(genpath('C:\Matlab Files\jsonlab'));
opt.type = 'train'; %{'train','test'}

% name = {'00.00', '00.01', '00.02', '00.03', '00.04', '00.05', ...
%     '00.06', '00.07', '00.08', '00.09', '00.10', '00.11', ...
%     '01.00', '01.01', '02.00', '02.01', '03.00', '04.00', '05.00'};
name = {'01.00', '01.01'};

for eid = 1:length(name)
    Exp_ID = name{eid};
    DirData_full = ['E:\NeuroFinder\',opt.type,' videos\neurofinder.',Exp_ID]; % folder of the original videos
    DirSave = ['E:\NeuroFinder\web\',opt.type,' videos\GT Masks']; % folder of the cropped videos
    if ~exist(DirSave)
        mkdir(DirSave)
    end

    % load the images
    files = dir(fullfile(DirData_full,'images\*.tiff'));

    fname = strcat(fullfile(DirData_full,'images\', files(1).name));
    img1 = imread(fname);
    [x, y] = size(img1);

    % load the regions (training data only)
    regions = loadjson(fullfile(DirData_full,'regions\regions.json'));
    FinalMasks = zeros(x, y,length(regions),'logical');

    for i = 1:length(regions)
        if isstruct(regions)
            coords = regions(i).coordinates+1;
        elseif iscell(regions)
            coords = regions{i}.coordinates+1;
        end
        Mask1 = zeros(x, y,'logical');
        Mask1(sub2ind([x, y], coords(:,1), coords(:,2))) = 1;
        FinalMasks(:,:,i) = Mask1;
    end

    % show the outputs
    figure();
%     subplot(1,2,1);
%     imagesc(mean(imgs,3)); colormap('gray'); axis image off;
%     subplot(1,2,2);
    imagesc(sum(FinalMasks,3)); colormap('gray'); axis image off;
    
    % save the GT Masks
    save(fullfile(DirSave,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
end