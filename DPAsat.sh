rm temp/* -rf
rm DPAsat/tem/add2tem/* -rf
python3 add2.py DPAsat/waitSolve/ DPAsat/tem/add2tem/
python3 python/psolvereal.py ../data/train/sr5 100 307 26 DPAsat//tem/add2tem/
for i in $(ls temp/gencnf/);do
      echo ""
      echo $i
	 ./hello ./temp/gencnf/$i 101 0.567 1
done


