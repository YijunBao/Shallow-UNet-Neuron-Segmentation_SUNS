REM REM Training pipeline
REM python train_CNN_params_noSF_J115.py
REM python train_CNN_params_noSF_J123.py
REM python train_CNN_params_noSF_K53.py
REM python train_CNN_params_noSF_YST.py

REM python "C:\Matlab Files\timer\timer_start_next.py"
REM Run SUNS batch
python test_batch_noSF_CM.py
REM Run SUNS online
python test_online_noSF_CM.py
REM Run SUNS batch
python test_batch_complete_CM.py
REM Run SUNS online
python test_online_complete_CM.py
REM python "C:\Matlab Files\timer\timer_stop_2.py"
