REM Generate sparse GT masks
python generate_sparse_GT.py

REM Training pipeline
python demo_train_CNN_params_multi_size.py

REM Run SUNS batch
python demo_test_batch_multi_size.py
REM Run SUNS online
python demo_test_online_multi_size.py
REM Run SUNS online with tracking
python demo_test_online_track_multi_size.py
