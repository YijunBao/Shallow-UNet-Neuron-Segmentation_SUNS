from preprocessing_functions import generate_masks_from_traces
# from preprocessing_functions_online import process_video, generate_masks


# %%
if __name__ == '__main__':
    list_exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    dir_video = 'D:\\ABO\\20 percent\\'
    dir_save = dir_video + 'ShallowUNet online\\noSF\\'  
    dir_GTMasks = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_'
    # sub_noise = '_noised_for3'
    # dir_parent = 'D:\\ABO\\20 percent\\resize_1.25\\'
    # dir_video = dir_parent + 'videos\\'
    # dir_GTMasks = dir_parent + 'GTMasks\\FinalMasks_FPremoved_'
    # dir_save = dir_parent + 'ShallowUNet\\complete\\'        
    list_thred_ratio = [5] # range(6,9) # 

    for Exp_ID in list_exp_ID: #[1:]
        file_mask = dir_GTMasks + Exp_ID + '.mat'
        generate_masks_from_traces(file_mask, list_thred_ratio, dir_save, Exp_ID)
        
