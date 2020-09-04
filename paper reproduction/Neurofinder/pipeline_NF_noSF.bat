REM Training pipeline
python train_CNN_params_NF_noSF.py

REM Run SUNS batch
python test_batch_NF_noSF.py
REM Run SUNS online
python test_online_NF_noSF.py
