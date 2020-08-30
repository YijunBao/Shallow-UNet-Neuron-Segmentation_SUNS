REM Prepare for FFT-based spatial filtering
python learn_wisdom_ABO.py
python learn_wisdom_2d_ABO.py


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
python test_batch_ABO_1to9_noSF_0.py
python test_batch_ABO_1to9_noSF_1.py
python test_batch_ABO_1to9_noSF_2.py
python test_batch_ABO_1to9_noSF_3.py
python test_batch_ABO_1to9_noSF_4.py
python test_batch_ABO_1to9_noSF_5.py
python test_batch_ABO_1to9_noSF_6.py
python test_batch_ABO_1to9_noSF_7.py
python test_batch_ABO_1to9_noSF_8.py
python test_batch_ABO_1to9_noSF_9.py
REM Run SUNS online. The post-processing will slow down after running many times, 
REM so I split each cross-validation into different scripts.
python test_online_ABO_1to9_noSF_0.py
python test_online_ABO_1to9_noSF_1.py
python test_online_ABO_1to9_noSF_2.py
python test_online_ABO_1to9_noSF_3.py
python test_online_ABO_1to9_noSF_4.py
python test_online_ABO_1to9_noSF_5.py
python test_online_ABO_1to9_noSF_6.py
python test_online_ABO_1to9_noSF_7.py
python test_online_ABO_1to9_noSF_8.py
python test_online_ABO_1to9_noSF_9.py


REM train on all 275 layer and test on 175 layer
REM Training pipeline
python train_CNN_params_ABO_10_noSF.py

REM Run SUNS batch
python test_batch_ABO_175_noSF.py
REM Run SUNS online
python test_online_ABO_175_noSF.py