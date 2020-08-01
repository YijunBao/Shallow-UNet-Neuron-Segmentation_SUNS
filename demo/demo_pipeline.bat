REM Training pipeline
REM Prepare for FFT-based spatial filtering
python demo_learn_wisdom.py
python demo_learn_wisdom_2d.py
REM Pre-processing and generate temporal labels
python demo_PreProcessing_masks.py
REM Train CNN
python demo_train.py
REM Search optimal post-processing parameters
python demo_inference_post_continue.py

REM Run SUNS batch
python demo_test_batch.py

REM Run SUNS online
python demo_test_online.py
