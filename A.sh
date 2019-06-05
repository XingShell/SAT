for i in $(ls temp/gencnf/*);do
       echo $i
	./CSCCSat $i 101
done
