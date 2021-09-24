#!/bin/sh
# Generate sparse GT masks
python generate_sparse_GT.py;

# Training pipeline
python demo_train_CNN_multi_size.py;
python demo_train_params_multi_size.py;

# Run SUNS batch
python demo_test_batch_multi_size.py;
# Run SUNS online
python demo_test_online_multi_size.py;
# Run SUNS online with tracking
python demo_test_online_track_multi_size.py;
