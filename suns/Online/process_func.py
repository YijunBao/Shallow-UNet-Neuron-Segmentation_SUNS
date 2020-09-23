from suns.PreProcessing.preprocessing_functions import median_calculation


def normalize_process(video_tf_past, med_frame2, med3_Array, dims_pad):
    med_frame3 = median_calculation(
        video_tf_past, med_frame2, dims_pad, 1, display=False)
    med3_Array[:] = med_frame3.ravel()