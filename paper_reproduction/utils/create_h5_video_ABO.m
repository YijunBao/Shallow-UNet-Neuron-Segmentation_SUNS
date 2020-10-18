% modified from
% 'https://github.com/soltanianzadeh/STNeuroNet/blob/master/demo_preprocess.m'
% Released under a GPL v2 license.

%%
layer = 275; % set the layer to be 175 um or 275 um.

%% Set directories
if layer == 275
    name = {'524691284', '531006860','502608215', '503109347','501484643', ...
            '501574836', '501729039', '539670003','510214538', '527048992'}; % name of the experiments
    DirData_full = 'D:\ABO'; % folder of the original videos
else % if layer == 175
    name = {'501271265', '501704220','501836392', '502115959', '502205092', ...
            '504637623', '510514474', '510517131','540684467', '545446482'}; % name of the experiments
    DirData_full = 'E:\ABO 175'; % folder of the original videos
end
DirData = fullfile(DirData_full,'20 percent'); % folder of the cropped videos

%% Set parameters. Border crop values are set as default.
for i = 1:numel(name)
    opt.ID = name{i};
    disp(opt.ID);
    dataFile = fullfile(DirData,[opt.ID,'.h5']); 
    if ~exist(dataFile)
        disp('Create cropped raw video');
        tic;
        vid = prepareAllen(opt, DirData_full, DirData);
        toc;
    end
end
