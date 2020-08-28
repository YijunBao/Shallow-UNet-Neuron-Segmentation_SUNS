REM REM Prepare for FFT-based spatial filtering
REM python learn_wisdom_ABO.py
REM python learn_wisdom_2d_ABO.py


REM leave-one-out cross validation
REM Training pipeline. The post-processing will slow down after running many times, 
REM so I split the parameter search into two scripts.
python train_CNN_params_ABO_complete.py
python train_params_ABO_continue_complete.py
echo off
REM REM Run SUNS batch
REM python test_batch_ABO_complete.py
REM REM Run SUNS online
REM python test_online_ABO_complete.py
REM REM Run SUNS online with tracking
REM python test_online_track_ABO_complete.py


REM REM train-1-test-9 cross validation
REM REM Training pipeline
REM python train_CNN_params_ABO_complete_1to9.py

REM REM Run SUNS batch. The post-processing will slow down after running many times, 
REM REM so I split each cross-validation into different scripts.
REM python test_batch_ABO_complete_1to9_0.py
REM python test_batch_ABO_complete_1to9_1.py
REM python test_batch_ABO_complete_1to9_2.py
REM python test_batch_ABO_complete_1to9_3.py
REM python test_batch_ABO_complete_1to9_4.py
REM python test_batch_ABO_complete_1to9_5.py
REM python test_batch_ABO_complete_1to9_6.py
REM python test_batch_ABO_complete_1to9_7.py
REM python test_batch_ABO_complete_1to9_8.py
REM python test_batch_ABO_complete_1to9_9.py
REM REM Run SUNS online. The post-processing will slow down after running many times, 
REM REM so I split each cross-validation into different scripts.
REM python test_online_ABO_complete_1to9_0.py
REM python test_online_ABO_complete_1to9_1.py
REM python test_online_ABO_complete_1to9_2.py
REM python test_online_ABO_complete_1to9_3.py
REM python test_online_ABO_complete_1to9_4.py
REM python test_online_ABO_complete_1to9_5.py
REM python test_online_ABO_complete_1to9_6.py
REM python test_online_ABO_complete_1to9_7.py
REM python test_online_ABO_complete_1to9_8.py
REM python test_online_ABO_complete_1to9_9.py


REM REM train on all 275 layer and test on 175 layer
REM REM Training pipeline
REM python train_CNN_params_ABO_complete_10.py

REM REM Run SUNS batch
REM python test_batch_ABO_complete_175.py
REM REM Run SUNS online
REM python test_online_ABO_complete_175.py

python "C:\Matlab Files\timer\timer_stop.py"
