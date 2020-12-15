REM python "C:\Matlab Files\timer\timer_start_next.py"
REM REM Training pipeline
REM python train_CNN_params_NF_noSF_2skip.py

REM REM Run SUNS batch
REM python test_batch_NF_noSF_2skip.py
REM REM Run SUNS online
REM python test_online_NF_noSF_2skip.py
REM python test_online_NF_noSF_2skip_update.py

REM python train_CNN_params_NF_noSF_vary_CNN.py 3 4 [1] elu True 1skip_1 0 noSF
REM python test_batch_NF_noSF_vary_CNN.py 3 4 [1] elu True 1skip_1 0 noSF
REM python test_online_NF_noSF_2skip.py
REM python test_online_NF_noSF_2skip_update.py
REM python train_CNN_params_NF_noSF_vary_CNN.py 3 4 [1] elu True 1skip_1 1 noSF_subtract
REM python test_batch_NF_noSF_vary_CNN.py 3 4 [1] elu True 1skip_1 1 noSF_subtract
REM python test_online_NF_noSF_2skip_copy.py
REM python test_online_NF_noSF_2skip_update_copy.py
REM python train_CNN_params_NF_noSF_vary_CNN.py 3 4 [1,2] elu True 2skip_1 0 noSF
REM python test_batch_NF_noSF_vary_CNN.py 3 4 [1,2] elu True 2skip_1 0 noSF
REM python test_online_NF_noSF_2skip.py
REM python test_online_NF_noSF_2skip_update.py
REM python train_CNN_params_NF_noSF_vary_CNN.py 3 4 [1,2] elu True 2skip_1 1 noSF_subtract
REM python test_batch_NF_noSF_vary_CNN.py 3 4 [1,2] elu True 2skip_1 1 noSF_subtract
REM python test_online_NF_noSF_2skip_copy.py
REM python test_online_NF_noSF_2skip_update_copy.py

REM python train_CNN_params_NF_All_noSF.py
REM python test_batch_NF_All_noSF.py
REM python test_online_NF_All_noSF.py
REM python test_online_NF_All_noSF.py
REM python train_CNN_params_NF_All_complete.py
REM python test_batch_NF_All_complete.py
python test_online_NF_All_complete.py
REM python test_online_NF_All_complete.py
REM python "C:\Matlab Files\timer\timer_stop.py"
