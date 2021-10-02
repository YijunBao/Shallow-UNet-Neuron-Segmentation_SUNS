#!/bin/sh
# Generate sparse GT masks
python generate_sparse_GT.py;

# Training pipeline
python demo_train_CNN.py;
python demo_train_params.py;

# Run SUNS batch
python demo_test_batch.py;
# Run SUNS online
python demo_test_online.py;
# Run SUNS online with tracking
python demo_test_online_track.py;
