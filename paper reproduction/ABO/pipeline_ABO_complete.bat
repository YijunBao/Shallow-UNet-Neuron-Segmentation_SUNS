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


REM REM train-1-test-9 cross validation
REM REM Training pipeline
REM python train_CNN_params_ABO_1to9_complete.py

REM Run SUNS batch. The post-processing will slow down after running many times, 
REM so I split each cross-validation into different scripts.
python test_batch_ABO_1to9_complete.py 0
python test_batch_ABO_1to9_complete.py 1
python test_batch_ABO_1to9_complete.py 2
python test_batch_ABO_1to9_complete.py 3
python test_batch_ABO_1to9_complete.py 4
python test_batch_ABO_1to9_complete.py 5
python test_batch_ABO_1to9_complete.py 6
python test_batch_ABO_1to9_complete.py 7
python test_batch_ABO_1to9_complete.py 8
python test_batch_ABO_1to9_complete.py 9

REM Run SUNS online. The post-processing will slow down after running many times, 
REM so I split each cross-validation into different scripts.
python test_online_ABO_1to9_complete.py 0
python test_online_ABO_1to9_complete.py 1
python test_online_ABO_1to9_complete.py 2
python test_online_ABO_1to9_complete.py 3
python test_online_ABO_1to9_complete.py 4
python test_online_ABO_1to9_complete.py 5
python test_online_ABO_1to9_complete.py 6
python test_online_ABO_1to9_complete.py 7
python test_online_ABO_1to9_complete.py 8
python test_online_ABO_1to9_complete.py 9


REM REM train on all 275 layer and test on 175 layer
REM REM Training pipeline
REM python train_CNN_params_ABO_10_complete.py
echo off
REM Run SUNS batch
python test_batch_ABO_175_complete.py
REM Run SUNS online
python test_online_ABO_175_complete.py

python "C:\Matlab Files\timer\timer_stop_2.py"
