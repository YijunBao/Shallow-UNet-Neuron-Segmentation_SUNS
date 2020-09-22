python "C:\Matlab Files\timer\timer_start_next_2.py"

python test_online_ABO_noSF.py
REM python test_online_ABO_complete.py
REM python test_online_track_ABO_noSF.py
REM python test_online_track_ABO_complete.py
python test_online_ABO_noSF_vary_merge.py 10
python test_online_ABO_noSF_vary_merge.py 20
REM python test_online_ABO_noSF_vary_merge.py 30
python test_online_ABO_noSF_vary_merge.py 50
python test_online_ABO_noSF_vary_merge.py 100
python test_online_ABO_noSF_vary_merge.py 200
python test_online_ABO_noSF_vary_merge.py 300
python test_online_ABO_noSF_vary_merge.py 500
python test_online_ABO_noSF_vary_merge.py 1000

python "C:\Matlab Files\timer\timer_stop.py"
