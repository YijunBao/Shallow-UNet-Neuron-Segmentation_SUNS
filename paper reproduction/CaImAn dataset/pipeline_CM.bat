python "C:\Matlab Files\timer\timer_start_next.py"
REM Training pipeline
python test_online_noSF_CM_update.py
python train_CNN_params_noSF_J115.py
python train_CNN_params_noSF_J123.py
python train_CNN_params_noSF_K53.py
python train_CNN_params_noSF_YST.py

REM Run SUNS batch
python test_batch_noSF_CM.py
REM Run SUNS online
python test_online_noSF_CM.py
python test_online_noSF_CM_update.py
python "C:\Matlab Files\timer\timer_stop.py"
