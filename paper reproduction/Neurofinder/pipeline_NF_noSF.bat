REM python "C:\Matlab Files\timer\timer_start_next.py"
REM REM Training pipeline
REM python train_CNN_params_NF_noSF.py

REM REM Run SUNS batch
REM python test_batch_NF_noSF.py
REM REM Run SUNS online
REM python test_online_NF_noSF.py

REM REM Training pipeline
REM python train_CNN_params_NF_noSF_transfer.py

REM Training pipeline
REM python train_CNN_params_NF_complete_transfer.py
REM Run SUNS batch
REM python test_batch_NF_complete_transfer.py
REM Run SUNS online
REM python test_online_NF_complete_transfer.py
REM python test_online_NF_complete.py

python test_batch_NF_noSF.py
python test_online_NF_noSF.py
python test_online_track_NF_noSF.py
python "C:\Matlab Files\timer\timer_stop.py"
