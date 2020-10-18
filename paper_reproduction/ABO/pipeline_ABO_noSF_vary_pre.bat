REM python "C:\Matlab Files\timer\timer_start_next.py"
REM python test_batch_ABO_noCNN.py 1 1 1 complete
REM python test_batch_ABO_noCNN.py 0 1 1 noSF
REM python test_batch_ABO_vary_pre.py 1 1 1 7 complete DL+100FL(1,0.25)
REM python test_batch_ABO_vary_pre.py 0 1 1 6 noSF DL+100FL(1,0.25)
REM python test_batch_ABO_vary_pre.py 1 0 1 7 noTF DL+100FL(1,0.25)
REM python test_batch_ABO_vary_pre.py 0 0 1 6 noSFTF DL+100FL(1,0.25)
REM python test_batch_ABO_vary_pre.py 1 1 1 7 expTF DL+100FL(1,0.25)
REM python test_batch_ABO_vary_pre.py 0 1 1 6 noSFexpTF DL+100FL(1,0.25)
REM python test_batch_ABO_vary_pre.py 0 1 0 6 noSFSNR DL
REM python test_batch_ABO_vary_pre.py 1 1 0 7 noSNR DL
python temporal_masks_ABO.py 0 4 noSF
python temporal_masks_ABO.py 0 8 noSF
python temporal_masks_ABO.py 0 3 noSF
REM python "C:\Matlab Files\timer\timer_stop_2.py"
