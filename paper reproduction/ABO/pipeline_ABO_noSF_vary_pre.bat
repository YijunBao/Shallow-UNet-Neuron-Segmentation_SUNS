REM python "C:\Matlab Files\timer\timer_start_next.py"
python test_batch_ABO_noCNN.py 1 1 1 complete
python test_batch_ABO_noCNN.py 0 1 1 noSF
python test_batch_ABO_vary_pre.py 1 1 1 7 complete DL+100FL(1,0.25)
python test_batch_ABO_vary_pre.py 0 1 1 6 noSF DL+100FL(1,0.25)
python test_batch_ABO_vary_pre.py 1 0 1 7 noTF DL+100FL(1,0.25)
python test_batch_ABO_vary_pre.py 0 0 1 6 noSFTF DL+100FL(1,0.25)
python test_batch_ABO_vary_pre.py 1 1 1 7 expTF DL+100FL(1,0.25)
python test_batch_ABO_vary_pre.py 0 1 1 6 noSFexpTF DL+100FL(1,0.25)
python test_batch_ABO_vary_pre.py 0 1 0 6 noSFSNR DL
python test_batch_ABO_vary_pre.py 1 1 0 7 noSNR DL

REM python "C:\Matlab Files\timer\timer_stop_2.py"
