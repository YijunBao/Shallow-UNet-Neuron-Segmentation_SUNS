REM python "C:\Matlab Files\timer\timer_start_next_2.py"
REM leave-one-out cross validation
REM Training pipeline. The post-processing will slow down after running many times, 
REM so I split the parameter search into two scripts.
python train_CNN_params_ABO_noSF.py
python train_params_ABO_continue_noSF.py

REM Run SUNS batch
python test_batch_ABO_noSF.py
REM Run SUNS online
python test_online_ABO_noSF.py
REM Run SUNS online with tracking
python test_online_track_ABO_noSF.py


REM train-1-test-9 cross validation
REM Training pipeline
python train_CNN_params_ABO_1to9_noSF.py

REM Run SUNS batch. The post-processing will slow down after running many times, 
REM so I split each cross-validation into different scripts.
python test_batch_ABO_1to9_noSF.py 0
python test_batch_ABO_1to9_noSF.py 1
python test_batch_ABO_1to9_noSF.py 2
python test_batch_ABO_1to9_noSF.py 3
python test_batch_ABO_1to9_noSF.py 4
python test_batch_ABO_1to9_noSF.py 5
python test_batch_ABO_1to9_noSF.py 6
python test_batch_ABO_1to9_noSF.py 7
python test_batch_ABO_1to9_noSF.py 8
python test_batch_ABO_1to9_noSF.py 9
REM Run SUNS online. The post-processing will slow down after running many times, 
REM so I split each cross-validation into different scripts.
python test_online_ABO_1to9_noSF.py 0
python test_online_ABO_1to9_noSF.py 1
python test_online_ABO_1to9_noSF.py 2
python test_online_ABO_1to9_noSF.py 3
python test_online_ABO_1to9_noSF.py 4
python test_online_ABO_1to9_noSF.py 5
python test_online_ABO_1to9_noSF.py 6
python test_online_ABO_1to9_noSF.py 7
python test_online_ABO_1to9_noSF.py 8
python test_online_ABO_1to9_noSF.py 9


REM train on all 275 layer and test on 175 layer
REM Training pipeline
python train_CNN_params_ABO_10_noSF.py

REM Run SUNS batch
python test_batch_ABO_175_noSF.py
REM Run SUNS online
python test_online_ABO_175_noSF.py
REM python "C:\Matlab Files\timer\timer_stop.py"
