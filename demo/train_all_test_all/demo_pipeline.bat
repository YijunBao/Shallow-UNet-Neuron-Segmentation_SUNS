REM Generate sparse GT masks
python generate_sparse_GT.py

REM Training pipeline
python demo_train_CNN_params.py

REM Run SUNS batch
python demo_test_batch.py
REM Run SUNS online
python demo_test_online.py
REM Run SUNS online with tracking
python demo_test_online_track.py
