% modified from
% 'https://github.com/soltanianzadeh/STNeuroNet/blob/master/demo_preprocess.m'
% Released under a GPL v2 license.

%% 
opt.type = 'test'; %{'train','test'}

name = {'100', '101','200', '201','400', '401'};
name_long = {'01.00', '01.01','02.00', '02.01','04.00', '04.01'};
DirData = ['E:\NeuroFinder\',opt.type,' videos\']; % folder of the original videos
DirData_full = ['E:\NeuroFinder\',opt.type,' videos\']; % folder of the cropped videos

if strcmp(opt.type, 'train')
    apd = '';
elseif strcmp(opt.type, 'test')
    apd = '.test';
end

%%
for i = 1:6
    opt.ID = name{i};
    disp(opt.ID);
    dataFile = [DirData,name_long{i},apd,'.h5'];
    if ~exist(dataFile)
        disp('Create cropped raw video');
        tic;
        vid = prepareNeurofinder(opt, DirData_full, DirData);
        toc;
    end
end

