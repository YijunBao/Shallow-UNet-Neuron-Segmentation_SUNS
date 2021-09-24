#!/bin/sh 
# Generate sparse GT masks
python generate_sparse_GT.py;

# Training pipeline
python demo_train_CNN_1to3.py;
python demo_train_params_1to3.py;

# Run SUNS batch
python demo_test_batch_1to3.py;
# Run SUNS online
python demo_test_online_1to3.py;
# Run SUNS online with tracking
python demo_test_online_track_1to3.py;