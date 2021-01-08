python "C:\Matlab Files\timer\timer_start_next.py"
python test_batch_ABO_simulation_noSF.py motion4
python test_batch_ABO_simulation_complete.py motion4
python test_online_ABO_simulation_noSF.py motion4
python test_online_ABO_simulation_complete.py motion4
python test_batch_ABO_simulation_noSF.py noise6
python test_batch_ABO_simulation_complete.py noise6
python test_online_ABO_simulation_noSF.py noise6
python test_online_ABO_simulation_complete.py noise6

python "C:\Matlab Files\timer\timer_stop_2.py"

@echo off
REM python test_online_ABO_noSF_vary_CNN.py 1skip_1 0 1
REM python test_online_ABO_noSF_vary_CNN.py 1skip_1 1 1
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 4 [1,2] elu True 2skip_1 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 4 [1,2] elu True 2skip_1 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 4 [1,2] elu True 2skip_1
REM python test_online_ABO_noSF_vary_CNN.py 2skip_1 0 1
REM python test_online_ABO_noSF_vary_CNN.py 2skip_1 1 1

REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 4 [1] elu True 1skip_1 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 4 [1] elu True 1skip_1 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 4 [1] elu True 1skip_1
REM python test_online_ABO_noSF_vary_CNN.py
REM python test_online_ABO_noSF_vary_CNN_update.py

REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 4 [1] elu True conv2d 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 4 [1] elu True conv2d 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 4 [1] elu True conv2d
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 4 [1,2] elu True conv2d_2skip 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 4 [1,2] elu True conv2d_2skip 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 4 [1,2] elu True conv2d_2skip
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 4 [] elu True conv2d_0skip 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 4 [] elu True conv2d_0skip 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 4 [] elu True conv2d_0skip
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 4 4 [1] elu True 1skip_481632 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 4 4 [1] elu True 1skip_481632 9
REM python test_batch_ABO_noSF_vary_CNN.py 4 4 [1] elu True 1skip_481632
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 4 4 [1,2] elu True 2skip_481632 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 4 4 [1,2] elu True 2skip_481632 9
REM python test_batch_ABO_noSF_vary_CNN.py 4 4 [1,2] elu True 2skip_481632
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 8 [1] elu True 1skip_81632 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 8 [1] elu True 1skip_81632 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 8 [1] elu True 1skip_81632
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 8 [1,2] elu True 2skip_81632 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 8 [1,2] elu True 2skip_81632 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 8 [1,2] elu True 2skip_81632
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 4 4 [1,2,3] elu True 3skip_481632 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 4 4 [1,2,3] elu True 3skip_481632 9
REM python test_batch_ABO_noSF_vary_CNN.py 4 4 [1,2,3] elu True 3skip_481632
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 4 [1,2] relu True 2skip_relu 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 4 [1,2] relu True 2skip_relu 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 4 [1,2] relu True 2skip_relu
REM python train_CNN_params_ABO_noSF_separable.py 4
REM python train_params_ABO_noSF_separable_continue.py 9
REM python test_batch_ABO_noSF_separable.py

REM python train_CNN_params_ABO_1to9_noSF.py 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 0 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 1 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 2 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 3 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 4 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 5 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 6 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 7 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 8 1skip_1to9
REM python test_batch_ABO_1to9_noSF.py 9 1skip_1to9
REM python train_CNN_params_ABO_1to9_noSF_subtract.py 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 0 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 1 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 2 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 3 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 4 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 5 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 6 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 7 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 8 1skip_1to9
REM python test_batch_ABO_1to9_noSF_subtract.py 9 1skip_1to9

REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 8 [1,2] elu False 2skip_888 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 8 [1,2] elu False 2skip_888 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 8 [1,2] elu False 2skip_888
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 16 [1,2] elu False 2skip_161616 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 16 [1,2] elu False 2skip_161616 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 16 [1,2] elu False 2skip_161616
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 4 2 [1,2] elu True 2skip_24816 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 4 2 [1,2] elu True 2skip_24816 9
REM python test_batch_ABO_noSF_vary_CNN.py 4 2 [1,2] elu True 2skip_24816
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 4 [1,2] elu False 2skip_444 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 4 [1,2] elu False 2skip_444 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 4 [1,2] elu False 2skip_444
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 2 4 [1,2] elu True 2skip_48 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 2 4 [1,2] elu True 2skip_48 9
REM python test_batch_ABO_noSF_vary_CNN.py 2 4 [1,2] elu True 2skip_48
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 2 8 [1,2] elu True 2skip_816 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 2 8 [1,2] elu True 2skip_816 9
REM python test_batch_ABO_noSF_vary_CNN.py 2 8 [1,2] elu True 2skip_816
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 2 16 [1,2] elu True 2skip_1632 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 2 16 [1,2] elu True 2skip_1632 9
REM python test_batch_ABO_noSF_vary_CNN.py 2 16 [1,2] elu True 2skip_1632
REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 3 2 [1,2] elu True 2skip_248 4
REM python train_CNN_params_ABO_noSF_vary_CNN_new_continue.py 3 2 [1,2] elu True 2skip_248 9
REM python test_batch_ABO_noSF_vary_CNN.py 3 2 [1,2] elu True 2skip_248

REM python train_CNN_params_ABO_noSF_vary_CNN_new.py 4 2 [1,2] elu True 2skip_24816 9
REM python test_batch_ABO_noSF_vary_CNN.py 4 2 [1,2] elu True 2skip_24816

REM python test_batch_ABO_noSF_heavy.py
REM python "C:\Matlab Files\timer\timer_stop_2.py"
