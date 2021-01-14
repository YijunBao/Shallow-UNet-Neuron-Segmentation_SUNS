%% Demo script that pre-processes an ABO data for training.
addpath(genpath('Software'))

%% Set directories
name = {'524691284', '531006860','502608215', '503109347','501484643', ...
        '501574836', '501729039', '539670003','510214538', '527048992'};
PreTime = zeros(10,1);

DirData_full = 'F:\ABO\';

% if ~exist(DirData,'dir')
%     mkdir(DirData);
% end

for i = 1:numel(name)
    opt.ID = name{i};
    opt.ds = 1;
    disp(opt.ID);
    for snr = 6 % [20, 40, 60]
        DirData = ['D:\ABO\20 percent\noise',num2str(snr),'\']; 
        dataFile = [DirData,opt.ID,'.h5']; %
        if ~exist(dataFile)
            vid = prepareAllen_noise(opt, DirData_full, DirData, snr);
        end
    end
end

