% modified from
% 'https://github.com/soltanianzadeh/STNeuroNet/blob/master/demo_preprocess.m'
% Released under a GPL v2 license.

%% 
opt.type = 'test'; %{'train','test'}

name = {'00.00', '00.01', '01.00', '01.01', '02.00', '02.01', '03.00', '04.00', '05.00'};
% name = {'00.00', '00.01', '00.02', '00.03', '00.04', '00.05', ...
%     '00.06', '00.07', '00.08', '00.09', '00.10', '00.11', ...
%     '01.00', '01.01', '02.00', '02.01', '03.00', '04.00', '05.00'};
DirData_full = ['E:\NeuroFinder\',opt.type,' videos\']; % folder of the original videos
DirSave = ['E:\NeuroFinder\web\',opt.type,' videos\']; % folder of the cropped videos

if strcmp(opt.type, 'train')
    apd = '';
elseif strcmp(opt.type, 'test')
    apd = '.test';
end

%%
for i = 1:length(name)
    opt.ID = name{i};
    disp(opt.ID);
    dataFile = fullfile(DirSave,opt.ID(1:2),[name{i},apd,'.h5']);
    if ~exist(dataFile)
        disp(['Create h5 raw video for ',opt.ID]);
        tic;
        vid = prepareNeurofinder(opt, DirData_full, fullfile(DirSave,opt.ID(1:2)));
        toc;
    end
end

