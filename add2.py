import os
import sys
'''增加两个纯文字
usage： python3 add2.py 源目录/ 目标目录/'''
if __name__=='__main__':
	solve_dir = sys.argv[1]
	trans_dir = sys.argv[2]
	filenames = [solve_dir + "/" + f for f in os.listdir(solve_dir)]
	for filename in filenames:
		with open(filename, 'r') as f:
			lines = f.readlines()
		i = 0
		while lines[i][0]!='p':
			i+=1		

		l = lines[i].split()
		v = int (l[-2])
		c = int (l[-1])
		l=lines[i].replace('p cnf %d %d'%(v,c),"p cnf %d %d"%(v+2,c+2))
		lines[i] = l
		Filename = filename.split('/')[-1]
		filenamenew = Filename.replace("v%dc%d"%(v,c),"v%dc%d"%(v+2,c+2))
		with open(trans_dir+filenamenew, 'w') as f1:
			f1.write(lines[i])
			f1.write("%d 0\n"%(v+1))
			f1.write("%d 0\n"%-(v+2))
			f1.writelines(lines[i+1:])
