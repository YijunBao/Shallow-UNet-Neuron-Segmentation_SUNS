REM Prepare for FFT-based spatial filtering
python learn_wisdom_CM.py
python learn_wisdom_2d_CM.py

REM Training pipeline
python train_CNN_params_complete_J115.py
python train_CNN_params_complete_J123.py
python train_CNN_params_complete_K53.py
python train_CNN_params_complete_YST.py
python train_CNN_params_noSF_J115.py
python train_CNN_params_noSF_J123.py
python train_CNN_params_noSF_K53.py
python train_CNN_params_noSF_YST.py

REM Run SUNS batch
python test_batch_complete_CM.py
python test_batch_noSF_CM.py
REM Run SUNS online
python test_online_complete_CM.py
python test_online_noSF_CM.py
