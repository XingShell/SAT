# -*- coding:UTF-8 -*-
import math
import numpy as np
import random
import datetime
import subprocess
import pickle
import sys
import os
import argparse
from options import add_neurosat_options
from neurosat import NeuroSAT
from newgenrate import  fun_generate
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
		answer = g.find_solutions(problem)
		c = np.where(answer, '1', '0') 
		filereadname = dir+problem.dimacs[0]
		if not os.path.exists("./temp/gencnf/"):
			os.mkdir("./temp/gencnf/")
		#filewritername = "./temp/gencnf/"+problem.dimacs[0]
		filewritername = "./temp/gencnf/%s"%problem.dimacs[0]
		readobject = open(filereadname, "r")
		writerobject = open(filewritername, "w")
		content = readobject.read()
		writerobject.write(content)
		#writerobject.write( ''.join([str(x) for x in np.where(c > 0, 1, 0)]))
		stringa = ''.join(c[0])
		print(stringa)
		writerobject.write( stringa)
		writerobject.close()
