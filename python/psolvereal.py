# -*- coding:UTF-8 -*-
import math
import numpy as np
import random
import datetime
import subprocess
import pickle
import sys
import os
import time
import argparse
from options import add_neurosat_options
from neurosat import NeuroSAT
from newgenrate import  fun_generate
#import seaborn as sns 
#import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
add_neurosat_options(parser)

parser.add_argument('solve_dir', action='store', type=str)
parser.add_argument('restore_id', action='store', type=int)
parser.add_argument('restore_epoch', action='store', type=int)
parser.add_argument('n_rounds', action='store', type=int)
parser.add_argument('dir', action='store', type=str)
opts = parser.parse_args()
setattr(opts, 'run_id', None)
setattr(opts, 'n_saves_to_keep', 1)
setattr(opts, 'n_rounds', 26)
setattr(opts, 'dir', None)

def distance(a,b):
	add = 0
	for i,_ in enumerate(a):
		add+=(a[i]-b[i])**2
	return add
g = NeuroSAT(opts)
g.restore()
dir = sys.argv[-1]
if not os.path.exists("./temp/real_pack/"):
	os.mkdir("./temp/real_pack/")
fun_generate(1,dir, "./temp/real_pack/")
# 这是一个数据格式包目录
opts.solve_dir = "./temp/real_pack/"
filenames = [opts.solve_dir + "/" + f for f in os.listdir(opts.solve_dir)]
num = 0
for filename in filenames:
	with open(filename, 'rb') as f:
		problems = pickle.load(f)
	for problem in problems:
		num += 1
		s = time.time()
		answer = g.find_psolutions(problem)
		e = time.time()
		print("time:",(e-s))
		TData_neg = []
		TData_pos = []
		for i in answer:
			#print(distance(i,answer[-1]),end=' ')
			TData_neg.append(distance(i,answer[-1]))
			TData_pos.append(distance(i,answer[-2]))
		ans = []
		for e,_ in enumerate(answer):
			ans.append(TData_neg[e]/(TData_pos[e]+TData_neg[e]))
		#print(ans)
		#plt.figure('Draw')
		#plt.hist(TData, bins=12, color=sns.desaturate("indianred", .8), alpha=.4)
		#plt.draw()  # 显示绘图
		#plt.pause(10)  
		
		#c = np.where(answer, '1', '0') 
		filereadname = dir+problem.dimacs[0]
		if not os.path.exists("./temp/gencnf/"):
			os.mkdir("./temp/gencnf/")
		#filewritername = "./temp/gencnf/"+problem.dimacs[0]
		filewritername = "./temp/gencnf/%s"%problem.dimacs[0]
		readobject = open(filereadname, "r")
		writerobject = open(filewritername, "w")
		content = readobject.readlines()
		writerobject.write(''.join(content[:]))
		#考虑!!!!!!!!!最后一行要不要
		#writerobject.write( ''.join([str(x) for x in np.where(c > 0, 1, 0)]))
		#stringa = ''.join(ans[0])
		#print(stringa)
		for i in ans[:len(ans)//2]:
			writerobject.write(str(i))
			writerobject.write(' ')
		writerobject.close()
