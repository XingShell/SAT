# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import tensorflow as tf
import random
import pickle
import argparse
import sys
from solver import solve_sat
from mk_problem import mk_batch_problem

def parse_dimacs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while lines[i].strip().split(" ")[0] == "c":
        i += 1
    header = lines[i].strip().split(" ")
    assert(header[0] == "p")
    n_vars = int(header[2])
    n_clauses = int(header[3])
    iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i+1:i+1+n_clauses]]
    answerpro =  np.random.rand(n_vars)
    if len(iclauses)-1 == n_clauses:
        ansprotem = [[float(s) for s in line.strip().split(" ")[:]] for line in lines[i+1+n_clauses:]]
        answerpro = []
        for line in ansprotem:
            answerpro.extend(line)
    return n_vars, iclauses, answerpro

def mk_dataset_filename_old(opts, n_batches):
    dimacs_path = opts.dimacs_dir.split("/")
    dimacs_dir = dimacs_path[-1] if dimacs_path[-1] != "" else dimacs_path[-2]
    return "%s/data_dir=%s__nb=%d.pkl" % (opts.out_dir, dimacs_dir, n_batches)

def mk_dataset_filename(opts, n_batches):
    #dimacs_path = opts.dimacs_dir.split("/")
    diamcs_path_string = "{"+opts.dimacs_dir+"}"
    diamcs_path_string = eval(repr(diamcs_path_string).replace('/', '@'))
    return "%s/data_dir=%s__nb=%d.pkl" % (opts.out_dir, diamcs_path_string, n_batches)

class Package:
    def __init__(self):
        self.dimacs_dir = "data_cnf"  # 输入cnf公式目录
        self.out_dir = "data_real_pack"
        self.max_nodes_per_batch = 1e100  # 文字+子句数量
        self.one = 0
        self.max_dimacs = None


def fun_generate(num=1,dir=None,dirout=None):
    opts = Package()
    problems = []
    batches = []
    n_nodes_in_batch = 0
    if dir != None:
        opts.dimacs_dir = dir
    if dirout!=None:
        opts.out_dir = dirout
    filenames = os.listdir(opts.dimacs_dir)
    filenames = sorted(filenames)
    prev_n_vars = None

    for filename in filenames:
        """   n_vars:变元数量，iclauses:[[1,2,-3],[?],[?]]
              n_clauses:子句数量， n_cells:文字和，n_nodes：2 * n_vars + n_clauses
        """
        print(filename)
        n_vars, iclauses, answerpro = parse_dimacs("%s/%s" % (opts.dimacs_dir, filename))
        print(n_vars)
        n_clauses = len(iclauses)
        n_cells = sum([len(iclause) for iclause in iclauses])
        n_nodes = 2 * n_vars + n_clauses
        if n_nodes > opts.max_nodes_per_batch:
            continue
        batch_ready = False
        opts.one = True  # 一个公式一个的包装
        if (opts.one and len(problems) > 0):
            batch_ready = True
        elif (prev_n_vars and n_vars != prev_n_vars):
            # bianyuan shuliang bu tong
            batch_ready = True
        elif (not opts.one) and n_nodes_in_batch + n_nodes > opts.max_nodes_per_batch:
            # chao guo dan yuan shu
            batch_ready = True
        if batch_ready:
            batches.append(mk_batch_problem(problems))
            print("batch %d done (%d vars, %d problems)...\n" % (len(batches), prev_n_vars, len(problems)))
            del problems[:]
            n_nodes_in_batch = 0
        prev_n_vars = n_vars
        # is_sat, stats = solve_sat(n_vars, iclauses)
        # 默认为可满足
        # print("----"+answerpro)
        problems.append((filename, n_vars, iclauses, True))
        n_nodes_in_batch += n_nodes

    if len(problems) > 0:
        batches.append(mk_batch_problem(problems))
        print("batch %d done (%d vars, %d problems)...\n" % (len(batches), n_vars, len(problems)))
        del problems[:]

    # create directory
    if not os.path.exists(opts.out_dir):
        os.mkdir(opts.out_dir)

    dataset_filename = mk_dataset_filename(opts, len(batches))
    print("Writing %d batches to %s...\n" % (len(batches), dataset_filename))
    with open(dataset_filename, 'wb') as f_dump:
       pickle.dump(batches, f_dump)

if __name__ == '__main__':
    """由于没有不满足的判断能力
    当没有参数：./data_cnf 作为.cnf目录
    参数 = 2：sys.argv[1] 作为.cnf目录
    参数 = 3：sys.argv[2] 输出打包格式目录"""
    if len(sys.argv) == 1:
        fun_generate()
    elif len(sys.argv) == 2:
        fun_generate(len(sys.argv), sys.argv[1])
    else:
        fun_generate(len(sys.argv), sys.argv[1], sys.argv[2])
