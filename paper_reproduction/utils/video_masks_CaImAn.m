% script modified from "matlab_read_data.m" in CaImAn dataset.

% script for loading data (inspired in neurofinder)
% first you need to unzip the images.zip files in the image subfolder for
% each dataset, then you can run this script to load the tiff
% requires one package from the matlab file exchange
%
% - jsonlab
% - http://www.mathworks.com/matlabcentral/fileexchange/33381-jsonlab--a-toolbox-to-encode-decode-json-files-in-matlab-octave

%%
clear;
dir_data_file = 'F:\CaImAn data\'; % The location of the unzipped files
list_caiman = { 'J115', 'J123', 'K53', 'YST'}; %, 'N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t'};
xyrange = [ 1, 224, 240, 463, 1, 224, 249, 472;
            1, 152 ,169, 320, 1, 216, 243, 458;
            1, 248, 265, 512, 1, 248, 265, 512;
            1,  88, 113, 200, 1, 120, 137, 256]; % lateral dimensions to crop four sub-videos.

%%
for ind=1:4
    %% find tiff files and order them
    data_name = list_caiman{ind};
    dir_data = fullfile(dir_data_file, ['images_',data_name]);
    zip_filename = fullfile(dir_data,'*.tif');
    xlsfiles = dir(zip_filename);
    xlsfiles = {xlsfiles.name};
    xlsfiles = sort(xlsfiles);
    numframes = numel(xlsfiles);
    if ~exist(fullfile(dir_data_file,data_name,'GT Masks'),'dir')
        mkdir(fullfile(dir_data_file,data_name,'GT Masks'))
    end

    %% load movies in the mov h5 file
    img = imread(fullfile(dir_data,xlsfiles{1}));
    [w, h] = size(img);
    t = numframes;
    type = class(img);
    for xpart = 1:2
        for ypart = 1:2
            xrange = xyrange(ind,2*xpart-1):xyrange(ind,2*xpart);
            yrange = xyrange(ind,2*ypart-1+4):xyrange(ind,2*ypart+4);
            fprintf('%s(%d:%d,%d:%d):  start',data_name, xyrange(ind,2*xpart-1),...
                xyrange(ind,2*xpart), xyrange(ind,2*ypart-1+4),xyrange(ind,2*ypart+4))
            mov = zeros(length(xrange),length(yrange),t,type);
            for counter = 1:numframes
                files = xlsfiles{counter};
                img = imread(fullfile(dir_data,files));
                mov(:,:,counter) = img(xrange,yrange);
                if mod(counter,100)==0
                    fprintf('\b\b\b\b\b\b%6d',counter) 
                end
            end
            fprintf('\n')
            h5_name = fullfile(dir_data_file,data_name,sprintf('%s_part%d%d.h5',data_name,xpart,ypart));
            if exist(h5_name,'file')
                delete(h5_name)
            end
            h5create(h5_name,'/mov',size(mov),'Datatype',type);
            h5write(h5_name,'/mov',mov);
        end
    end
    
    %% load the regions (training data only)
    regions = jsondecode(fileread(fullfile(dir_data_file,...
        'WEBSITE_basic',data_name,'regions','consensus_regions.json')));
    num_masks = length(regions);
    mask = zeros(w, h, 'logical');
    masks = zeros(w, h, num_masks, 'logical');

    for i = 1:num_masks
        if isstruct(regions)
            coords = regions(i).coordinates+2;
        elseif iscell(regions)
            coords = regions{i}.coordinates+2;
        end
        mask = zeros(w, h, 'logical');
        mask(sub2ind([w, h], coords(:,1), coords(:,2))) = 1;
        masks(:,:,i)=mask;
    end
    areas = squeeze(sum(sum(masks,1),2));
    
    for xpart = 1:2
        for ypart = 1:2
            xrange = xyrange(ind,2*xpart-1):xyrange(ind,2*xpart);
            yrange = xyrange(ind,2*ypart-1+4):xyrange(ind,2*ypart+4);
            FinalMasks = masks(xrange,yrange,:);
            areas_cut = squeeze(sum(sum(FinalMasks,1),2));
            areas_ratio = areas_cut./areas;
            FinalMasks(:,:,areas_ratio<1/3)=[];
            mask_name = fullfile(dir_data_file,data_name,'GT Masks',sprintf('FinalMasks_%s_part%d%d.mat',data_name,xpart,ypart));
            save(mask_name,'FinalMasks','-v7.3');
        end
    end
end