% modified from
% 'https://github.com/soltanianzadeh/STNeuroNet/blob/master/demo_preprocess.m'
% Released under a GPL v2 license.

%% 
opt.type = 'test'; %{'train','test'}

name = {'100', '101', '200', '201', '400', '401'}; %
name_long = {'01.00', '01.01', '02.00', '02.01', '04.00', '04.01'};
DirData_full = ['E:\NeuroFinder\',opt.type,' videos\']; % folder of the original videos
DirData = ['E:\NeuroFinder\bin3\',opt.type,' videos\']; % folder of the cropped videos
list_ds = [3,3,3,3,2,1];

if strcmp(opt.type, 'train')
    apd = '';
elseif strcmp(opt.type, 'test')
    apd = '.test';
end

%%
for i = 1:length(name)
    opt.ID = name{i};
    disp(opt.ID);
    opt.ds = list_ds(ind);
    dataFile = fullfile(DirData,[name_long{i},apd,'.h5']);
    if ~exist(dataFile)
        disp('Create cropped raw video');
        tic;
        vid = prepareNeurofinder_bin(opt, DirData_full, DirData);
        toc;
    end
end
