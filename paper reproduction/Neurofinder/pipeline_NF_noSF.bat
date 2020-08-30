python "C:\Matlab Files\timer\timer_start_next_2.py"
REM REM Prepare for FFT-based spatial filtering
REM python learn_wisdom_NF.py
REM python learn_wisdom_2d_NF.py

REM Training pipeline
python train_CNN_params_NF_noSF.py

REM Run SUNS batch
python test_batch_NF_noSF.py
REM Run SUNS online
python test_online_NF_noSF.py
