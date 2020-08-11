REM REM Training pipeline
REM REM Prepare for FFT-based spatial filtering
REM python demo_learn_wisdom.py
REM python demo_learn_wisdom_2d.py
REM REM Pre-processing and generate temporal labels
REM python demo_PreProcessing_masks.py
REM REM Train CNN
REM python demo_train.py
REM REM Search optimal post-processing parameters
python demo_inference_post_continue.py

REM Run SUNS batch
python demo_test_batch.py

REM Run SUNS online
python demo_test_online.py
