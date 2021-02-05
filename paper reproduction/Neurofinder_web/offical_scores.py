import os
import json
from neurofinder import load, match, centers, shapes
from scipy.io import savemat, loadmat

def evaluate(files, threshold=5):
    a = load(files[0])
    b = load(files[1])

    recall, precision = centers(a, b, threshold=threshold)
    inclusion, exclusion = shapes(a, b, threshold=threshold)

    if recall == 0 and precision == 0:
        combined = 0
    else:
        combined = 2 * (recall * precision) / (recall + precision)
    
    result = {'combined': round(combined, 4), 'inclusion': round(inclusion, 4), 'precision': round(precision, 4), 'recall': round(recall, 4), 'exclusion': round(exclusion, 4)}
    return result


if __name__ == '__main__':
    list_Exp_ID_full = [['00.00', '00.01', '00.02', '00.03', '00.04', '00.05', \
                        '00.06', '00.07', '00.08', '00.09', '00.10', '00.11'], \
                        ['01.00', '01.01'], ['02.00', '02.01'], ['03.00'], ['04.00'], ['05.00']] 
                        # '04.01' is renamed as '05.00', because the imaging condition is different from '04.00'

    for ind_set in [0,1,2,3,4,5]: # [5]: # 
        list_Exp_ID = list_Exp_ID_full[ind_set]
        dir_origin = 'E:\\NeuroFinder\\train videos\\neurofinder.'
        dir_video = 'E:\\NeuroFinder\\web\\train videos\\' + list_Exp_ID[0][:2]
        dir_output = dir_video + '\\noSF\\trial 1\\output_masks track\\'

        nvideo = len(list_Exp_ID)
        list_CV = list(range(0,nvideo))
        Output_Info_All = loadmat(dir_output + 'Output_Info_All.mat')
        list_Recall = Output_Info_All['list_Recall']
        list_Precision = Output_Info_All['list_Precision']
        list_F1 = Output_Info_All['list_F1']
        list_time = Output_Info_All['list_time']
        list_time_frame = Output_Info_All['list_time_frame']

        for CV in list_CV:
            Exp_ID = list_Exp_ID[CV]
            dir_GTMasks = dir_origin + Exp_ID + '\\regions\\regions.json'
            dir_output_masks = dir_output + 'Output_Masks_' + Exp_ID + '.json'
            result = evaluate([dir_GTMasks, dir_output_masks])
            list_Recall[CV] = result['recall']
            list_Precision[CV] = result['precision']
            list_F1[CV] = result['combined']
            # print(Exp_ID, result)
        
        Info_dict = {'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 
            'list_time':list_time, 'list_time_frame':list_time_frame}
        savemat(dir_output + 'Output_Info_All_offical.mat', Info_dict)
