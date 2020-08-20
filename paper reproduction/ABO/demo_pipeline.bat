REM Prepare for FFT-based spatial filtering
python learn_wisdom_ABO.py
python learn_wisdom_2d_ABO.py


REM leave-one-out cross validation
REM Training pipeline
python train_CNN_params_ABO_complete.py
python train_CNN_params_ABO_noSF.py

REM Run SUNS batch
python test_batch_ABO_complete.py
python test_batch_ABO_noSF.py
REM Run SUNS online
python test_online_ABO_complete.py
python test_online_ABO_noSF.py
REM Run SUNS online with tracking
python test_online_track_ABO_complete.py
python test_online_track_ABO_noSF.py


REM train-1-test-9 cross validation
REM Training pipeline
python train_CNN_params_ABO_complete_1to9.py
python train_CNN_params_ABO_noSF_1to9.py

REM Run SUNS batch
python test_batch_ABO_complete_1to9.py
python test_batch_ABO_noSF_1to9.py
REM Run SUNS online
python test_online_ABO_complete_1to9.py
python test_online_ABO_noSF_1to9.py


REM train on all 275 layer and test on 175 layer
REM Training pipeline
python train_CNN_params_ABO_complete_10.py
python train_CNN_params_ABO_noSF_10.py

REM Run SUNS batch
python test_batch_ABO_complete_175.py
python test_batch_ABO_noSF_175.py
REM Run SUNS online
python test_online_ABO_complete_175.py
python test_online_ABO_noSF_175.py