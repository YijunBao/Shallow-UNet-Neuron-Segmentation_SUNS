#!/bin/sh
# # Prepare for FFT-based spatial filtering
# # This is not needed in the demo, 
# # because the demo does not use spatial filtering.
# python demo_learn_wisdom.py;
# python demo_learn_wisdom_2d.py;

# Training pipeline
python demo_train_CNN.py;
python demo_train_params.py;

# Run SUNS batch
python demo_test_batch.py;
# Run SUNS online
python demo_test_online.py;
# Run SUNS online with tracking
python demo_test_online_track.py;
