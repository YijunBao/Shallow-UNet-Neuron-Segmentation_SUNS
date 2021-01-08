REM REM Training pipeline
REM python train_CNN_params_NF_noSF.py

REM REM Run SUNS batch
REM python test_batch_NF_noSF.py
REM REM Run SUNS online
REM python test_online_NF_noSF.py

REM REM Training pipeline
REM python train_CNN_params_NF_noSF_transfer.py

REM REM Run SUNS batch
REM python test_batch_NF_noSF_transfer.py
REM REM Run SUNS online
REM python test_online_NF_noSF_transfer.py
python test_online_NF_complete.py
REM Training pipeline
python train_CNN_params_NF_complete_transfer.py
REM Run SUNS batch
python test_batch_NF_complete_transfer.py
REM Run SUNS online
python test_online_NF_complete_transfer.py
python test_online_NF_complete.py

python train_CNN_params_NF_noSFexpTF.py
python test_batch_NF_noSFexpTF.py
python test_online_NF_noSFexpTF.py
python "C:\Matlab Files\timer\timer_stop.py"
