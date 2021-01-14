%% Demo script that pre-processes an ABO data for training.
addpath(genpath('Software'))

%% Set directories
name = {'524691284', '531006860','502608215', '503109347','501484643', ...
        '501574836', '501729039', '539670003','510214538', '527048992'};
PreTime = zeros(10,1);
fs = 30; % Hz
f_lpf = 10; % Hz

DirData_full = 'F:\ABO\';

% if ~exist(DirData,'dir')
%     mkdir(DirData);
% end

for i = 1:numel(name)
    opt.ID = name{i};
    opt.ds = 1;
    disp(opt.ID);
    for sigma = 4
        DirData = ['D:\ABO\20 percent\motion',num2str(sigma),'\']; 
        dataFile = [DirData,opt.ID,'.h5']; %
        if ~exist(dataFile)
            vid = prepareAllen_motion(opt, DirData_full, DirData, sigma,fs,f_lpf);
        end
    end
end

