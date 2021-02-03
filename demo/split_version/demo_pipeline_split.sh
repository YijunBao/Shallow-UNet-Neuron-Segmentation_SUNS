#!/bin/sh
# Training pipeline
python demo_train_CNN.py;
python demo_train_params.py;

# Run SUNS batch
python demo_test_batch.py;
# Run SUNS online
python demo_test_online.py;
# Run SUNS online with tracking
python demo_test_online_track.py;
