REM REM Prepare for FFT-based spatial filtering
REM python learn_wisdom_ABO.py
REM python learn_wisdom_2d_ABO.py
REM python "C:\Matlab Files\timer\timer_start_next.py"

REM REM leave-one-out cross validation
REM REM Training pipeline. The post-processing will slow down after running many times, 
REM REM so I split the parameter search into two scripts.
REM python train_CNN_params_ABO_complete.py
REM python train_params_ABO_continue_complete.py

REM REM Run SUNS batch
REM python test_batch_ABO_complete.py
REM REM Run SUNS online
REM python test_online_ABO_complete.py
REM REM Run SUNS online with tracking
REM python test_online_track_ABO_complete.py


REM train-1-test-9 cross validation
REM Training pipeline
python train_CNN_params_ABO_1to9_complete.py
echo off
REM REM Run SUNS batch. The post-processing will slow down after running many times, 
REM REM so I split each cross-validation into different scripts.
python test_batch_ABO_1to9_complete_0.py
python test_batch_ABO_1to9_complete_1.py
python test_batch_ABO_1to9_complete_2.py
REM python test_batch_ABO_1to9_complete_3.py
REM python test_batch_ABO_1to9_complete_4.py
REM python test_batch_ABO_1to9_complete_5.py
REM python test_batch_ABO_1to9_complete_6.py
REM python test_batch_ABO_1to9_complete_7.py
REM python test_batch_ABO_1to9_complete_8.py
REM python test_batch_ABO_1to9_complete_9.py

REM REM Run SUNS online. The post-processing will slow down after running many times, 
REM REM so I split each cross-validation into different scripts.
REM python test_online_ABO_1to9_complete_0.py
REM python test_online_ABO_1to9_complete_1.py
REM python test_online_ABO_1to9_complete_2.py
REM python test_online_ABO_1to9_complete_3.py
REM python test_online_ABO_1to9_complete_4.py
REM python test_online_ABO_1to9_complete_5.py
REM python test_online_ABO_1to9_complete_6.py
REM python test_online_ABO_1to9_complete_7.py
REM python test_online_ABO_1to9_complete_8.py
REM python test_online_ABO_1to9_complete_9.py


REM train on all 275 layer and test on 175 layer
REM Training pipeline
python train_CNN_params_ABO_10_complete_10.py

REM REM Run SUNS batch
REM python test_batch_ABO_175_complete_175.py
REM REM Run SUNS online
REM python test_online_ABO_175_complete_175.py

REM python "C:\Matlab Files\timer\timer_stop_2.py"
