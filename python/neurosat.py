# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import math
import random
import os
import time
from confusion import ConfusionMatrix
from problems_loader import init_problems_loader
from mlp import MLP
from util import repeat_end, decode_final_reducer, decode_transfer_fn
from tensorflow.contrib.rnn import LSTMStateTuple
from sklearn.cluster import KMeans
from tensorflow.python.ops import gen_nn_ops
dim = 256
endfn = 16*dim*1//4
class NeuroSAT(object):
    def __init__(self, opts):
        self.opts = opts
        self.batchnum = 1
        self.final_reducer = decode_final_reducer(opts.final_reducer)

        self.build_network()
        self.train_problems_loader = None

    def init_random_seeds(self):
        tf.set_random_seed(self.opts.tf_seed)
        np.random.seed(self.opts.np_seed)

    def construct_session(self):
        self.sess = tf.Session()

    def declare_parameters(self):
        opts = self.opts
        with tf.variable_scope('params') as scope:
            self.L_init = tf.get_variable(name="L_init", initializer=tf.random_normal([1, self.opts.d]))
            self.C_init = tf.get_variable(name="C_init", initializer=tf.random_normal([1, self.opts.d]))

            self.LC_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("LC_msg"))
            self.CL_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("CL_msg"))

            self.L_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))
            self.C_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))

            self.L_vote = MLP(opts, endfn, repeat_end(opts.d, opts.n_vote_layers, 1), name=("L_vote"))
            self.vote_bias = tf.get_variable(name="vote_bias", shape=[], initializer=tf.zeros_initializer())

    def declare_placeholders(self):
        self.n_vars = tf.placeholder(tf.int32, shape=[], name='n_vars')
        self.n_lits = tf.placeholder(tf.int32, shape=[], name='n_lits')
        self.n_clauses = tf.placeholder(tf.int32, shape=[], name='n_clauses')

        self.L_unpack = tf.sparse_placeholder(tf.float32, shape=[None, None], name='L_unpack')
        self.is_sat = tf.placeholder(tf.bool, shape=[None], name='is_sat')

        # useful helpers
        self.n_batches = tf.shape(self.is_sat)[0]
        self.n_vars_per_batch = tf.div(self.n_vars, self.n_batches)

    def while_cond(self, i, L_state, C_state):
        return tf.less(i, self.opts.n_rounds)

    def flip(self, lits):
        return tf.concat([lits[self.n_vars:(2*self.n_vars), :], lits[0:self.n_vars, :]], axis=0)

    def while_body(self, i, L_state, C_state):
        LC_pre_msgs = self.LC_msg.forward(L_state.h)
        LC_msgs = tf.sparse_tensor_dense_matmul(self.L_unpack, LC_pre_msgs, adjoint_a=True)

        with tf.variable_scope('C_update') as scope:
            _, C_state = self.C_update(inputs=LC_msgs, state=C_state)

        CL_pre_msgs = self.CL_msg.forward(C_state.h)
        CL_msgs = tf.sparse_tensor_dense_matmul(self.L_unpack, CL_pre_msgs)

        with tf.variable_scope('L_update') as scope:
            _, L_state = self.L_update(inputs=tf.concat([CL_msgs, self.flip(L_state.h)], axis=1), state=L_state)

        return i+1, L_state, C_state

    def pass_messages(self):
        with tf.name_scope('pass_messages') as scope:
            denom = tf.sqrt(tf.cast(self.opts.d, tf.float32))

            L_output = tf.tile(tf.div(self.L_init, denom), [self.n_lits, 1])
            C_output = tf.tile(tf.div(self.C_init, denom), [self.n_clauses, 1])

            L_state = LSTMStateTuple(h=L_output, c=tf.zeros([self.n_lits, self.opts.d]))
            C_state = LSTMStateTuple(h=C_output, c=tf.zeros([self.n_clauses, self.opts.d]))

            _, L_state, C_state = tf.while_loop(self.while_cond, self.while_body, [0, L_state, C_state])

        self.final_lits = L_state.h
        self.final_clauses = C_state.h

    def spp_layer(input_, levels=4, name = 'SPP_layer',pool_type = 'max_pool'):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(name):
            for l in range(levels):
                l = l + 1
                ksize = [1, np.ceil(shape[1]/ l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]          
                strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]
                if pool_type == 'max_pool':
                    pool = tf.nn.max_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                    pool = tf.reshape(pool,(shape[0],-1),)  
                else :
                    pool = tf.nn.avg_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                    pool = tf.reshape(pool,(shape[0],-1))
            #print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
                if l == 1:
                    x_flatten = tf.reshape(pool,(shape[0],-1))
                else:
                    x_flatten = tf.concat((x_flatten,pool),axis=1)
            #print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
            # pool_outputs.append(tf.reshape(pool, [tf.shape(pool)[1], -1]))
        return x_flatten


    def weight_variable(self, shape):
        # 这里是构建初始变量
        initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
        # 创建变量
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 在这里定义残差网络的id_block块，此时输入和输出维度相同
    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block):
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            # first
            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            b_conv1 = self.bias_variable([f1])
            X = tf.nn.relu(X + b_conv1)

            # second
            W_conv2 = self.weight_variable([1, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            b_conv2 = self.bias_variable([f2])
            X = tf.nn.relu(X + b_conv2)

            # third

            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            b_conv3 = self.bias_variable([f3])
            X = tf.nn.relu(X + b_conv3)
            # final step
            add = tf.add(X, X_shortcut)
            # b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add)

        return add_result

    # 这里定义conv_block模块，由于该模块定义时输入和输出尺度不同，故需要进行卷积操作来改变尺度，从而得以相加
    def convolutional_block(self, X_input, kernel_size, in_filter,out_filters, stage, block, stride=2):
        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            # first
            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, stride, 1], padding='SAME')
            b_conv1 = self.bias_variable([f1])
            X = tf.nn.relu(X + b_conv1)

            # second
            W_conv2 = self.weight_variable([1, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            b_conv2 = self.bias_variable([f2])
            X = tf.nn.relu(X + b_conv2)

            # third
            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            b_conv3 = self.bias_variable([f3])
            X = tf.nn.relu(X + b_conv3)
            # shortcut path
            W_shortcut = self.weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, 1, stride, 1], padding='VALID')

            # final
            add = tf.add(x_shortcut, X)
            # 建立最后融合的权重
            # b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add)

        return add_result

    def compute_logits(self):
        with tf.name_scope('compute_logits') as scope:
            t1x = tf.reshape(self.final_lits[0:self.n_vars], [self.n_batches, self.n_vars_per_batch, self.opts.d, 1])
            t2x = tf.reshape(self.final_lits[self.n_vars:], [self.n_batches, self.n_vars_per_batch, self.opts.d, 1])
            self.image = tf.concat([t1x, t2x], axis=3)
            self.image = tf.reshape(self.image, [self.n_batches, -1,  self.opts.d, 2])

            w_conv1 = self.weight_variable([1, 2, 2, 4])
            x = tf.nn.conv2d(self.image, w_conv1, strides=[1, 1, 2, 1], padding='SAME')
            b_conv1 = self. bias_variable([4])
            x = tf.nn.relu(x + b_conv1)

            x = tf.nn.max_pool(x, ksize=[1, 1, 3, 1],
                               strides=[1, 1, 1, 1], padding='SAME')
            # stage 2
            x = self.convolutional_block(X_input=x, kernel_size=2, in_filter=4, out_filters=[4, 8, 16], stage=2,
                                    block='a', stride=1)
            x = self.convolutional_block(X_input=x, kernel_size=3, in_filter=16, out_filters=[16, 20, 32], stage=2,
                                    block='a', stride=1)
            x = self.identity_block(x, 3, 32, [8, 16, 32], stage=2, block='b')
            x = self.identity_block(x, 3, 32, [8, 16, 32], stage=2, block='c')
            # 上述操作后张量尺寸变成1x32x16
            x = self.convolutional_block(X_input=x, kernel_size=3, in_filter=32, out_filters=[25, 20, 16], stage=2,
                                    block='a', stride=1)
            x = tf.nn.max_pool(x, [1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

            self.feature_map_out = tf.reshape(x, (-1, 16*dim*1//4))
            self.all_votes = self.L_vote.forward(self.feature_map_out) + self.vote_bias
            # self.all_votes = tf.reshape(self.all_votes, [self.n_batches, -1])
            self.all_votes_batched = tf.reshape(self.all_votes, [self.n_batches, self.n_vars_per_batch, 1])
            self.logits = self.final_reducer(self.all_votes_batched)
            

    def compute_cost(self):
        self.predict_costs = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.is_sat, tf.float32))
        self.predict_cost = tf.reduce_mean(self.predict_costs)
        '''with tf.name_scope('l2') as scope:
            l2_cost = tf.zeros([])
            for var in tf.trainable_variables():
                l2_cost += tf.nn.l2_loss(var)
        self.cost = tf.identity(self.predict_cost + self.opts.l2_weight * l2_cost, name="cost")'''
        self.cost = tf.identity(self.predict_cost, name="cost")
        tf.summary.scalar('cost', self.cost)
        with tf.name_scope('correct_prediction'):
            one = (tf.cast(self.is_sat, tf.int32)+1)/(tf.cast(self.is_sat, tf.int32)+1)
            compare = self.logits > 0
            compare = tf.where(compare, one, 0*one)
            self.correct_prediction = tf.equal(tf.cast(compare, tf.int32), tf.cast(self.is_sat, tf.int32))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train/logs', self.sess.graph)
    def build_optimizer(self):
        opts = self.opts
        self.global_step = tf.get_variable("global_step", shape=[], initializer=tf.zeros_initializer(), trainable=False)
        if opts.lr_decay_type == "no_decay":
            self.learning_rate = tf.constant(opts.lr_start)
        elif opts.lr_decay_type == "poly":
            self.learning_rate = tf.train.polynomial_decay(opts.lr_start, self.global_step, opts.lr_decay_steps, opts.lr_end, power=opts.lr_power)
        elif opts.lr_decay_type == "exp":
            self.learning_rate = tf.train.exponential_decay(opts.lr_start, self.global_step, opts.lr_decay_steps, opts.lr_decay, staircase=False)
        else:
            raise Exception("lr_decay_type must be 'no_decay', 'poly' or 'exp'")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, self.opts.clip_val)
        self.apply_gradients = optimizer.apply_gradients(zip(gradients, variables), name='apply_gradients', global_step=self.global_step)

    def initialize_vars(self):
        tf.global_variables_initializer().run(session=self.sess)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.opts.n_saves_to_keep)
        if self.opts.run_id:
            self.save_dir = "snapshots/run%d" % self.opts.run_id
            self.save_prefix = "%s/snap" % self.save_dir

    def build_network(self):
        self.init_random_seeds()
        self.construct_session()
        self.declare_parameters()
        self.declare_placeholders()
        self.pass_messages()
        self.compute_logits()
        self.compute_cost()
        self.build_optimizer()
        self.initialize_vars()
        self.init_saver()

    def save(self, epoch):
        self.saver.save(self.sess, self.save_prefix, global_step=epoch)

    def restore(self):
        snapshot = "snapshots/run%d/snap-%d" % (self.opts.restore_id, self.opts.restore_epoch)
        self.saver.restore(self.sess, snapshot)

    def build_feed_dict(self, problem):
        d = {}
        d[self.n_vars] = problem.n_vars
        d[self.n_lits] = problem.n_lits
        d[self.n_clauses] = problem.n_clauses
        d[self.L_unpack] = tf.SparseTensorValue(indices=problem.L_unpack_indices,
                                                 values=np.ones(problem.L_unpack_indices.shape[0]),
                                                 dense_shape=[problem.n_lits, problem.n_clauses])

        d[self.is_sat] = problem.is_sat
        return d

    def train_epoch(self, epoch):
        if self.train_problems_loader is None:
            self.train_problems_loader = init_problems_loader(self.opts.train_dir)
        epoch_start = time.clock()
        epoch_train_cost = 0.0
        epoch_train_mat = ConfusionMatrix()
        train_problems, train_filename = self.train_problems_loader.get_next()

        for problem in train_problems:
            d = self.build_feed_dict(problem)
            self.batchnum = problem.n_vars//len(problem.is_sat)
            acc,feature_map_out, _, logits, cost, image, issat, XX = self.sess.run(
                 [self.accuracy, self.all_votes, self.apply_gradients, self.logits, self.cost, self.image, self.is_sat, self.merged], feed_dict=d)
            epoch_train_cost += cost
            epoch_train_mat.update(problem.is_sat, logits > 0)
            # print(acc)
        epoch_train_cost /= len(train_problems)
        epoch_train_mat = epoch_train_mat.get_percentages()
        epoch_end = time.clock()
        learning_rate = self.sess.run(self.learning_rate)
        self.save(epoch)
        self.train_writer.add_summary(XX, epoch)
        return (train_filename, epoch_train_cost, epoch_train_mat, learning_rate, epoch_end - epoch_start)

    def test(self, test_data_dir):
        test_problems_loader = init_problems_loader(test_data_dir)
        results = []

        while test_problems_loader.has_next():
            test_problems, test_filename = test_problems_loader.get_next()

            epoch_test_cost = 0.0
            epoch_test_mat = ConfusionMatrix()

            for problem in test_problems:
                d = self.build_feed_dict(problem)
                logits, cost = self.sess.run([self.logits, self.cost], feed_dict=d)
                epoch_test_cost += cost
                epoch_test_mat.update(problem.is_sat, logits > 0)

            epoch_test_cost /= len(test_problems)
            epoch_test_mat = epoch_test_mat.get_percentages()

            results.append((test_filename, epoch_test_cost, epoch_test_mat))

        return results

    def find_solutionsold(self, problem):
        def flip_vlit(vlit):
            if vlit < problem.n_vars: return vlit + problem.n_vars
            else: return vlit - problem.n_vars
        n_batches = len(problem.is_sat)
        n_vars_per_batch = problem.n_vars // n_batches

        d = self.build_feed_dict(problem)
        all_votes, final_lits, logits, costs = self.sess.run([self.all_votes, self.final_lits, self.logits, self.predict_costs], feed_dict=d)

        solutions = []
        for batch in range(len(problem.is_sat)):
            # decode_cheap_A = (lambda vlit: all_votes[vlit, 0] > all_votes[flip_vlit(vlit), 0])
            # decode_cheap_B = (lambda vlit: not decode_cheap_A(vlit))

            def reify(phi):
                xs = list(zip([phi(vlit) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)],
                              [phi(flip_vlit(vlit)) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)]))
                def one_of(a, b): return (a and (not b)) or (b and (not a))
                assert(all([one_of(x[0], x[1]) for x in xs]))
                return [x[0] for x in xs]

            # if self.solves(problem, batch, decode_cheap_A): solutions.append(reify(decode_cheap_A))
            # elif self.solves(problem, batch, decode_cheap_B): solutions.append(reify(decode_cheap_B))
            # else:

            L = np.reshape(final_lits, [2 * n_batches, n_vars_per_batch, self.opts.d])
            # L = np.concatenate([L[batch, :, :], L[n_batches + batch, :, :]], axis=0)
            # print(L.shape)
            L1 = L[batch, :, :]
            kmeans = KMeans(n_clusters=2, random_state=0).fit(L1)
            distances = kmeans.transform(L1)
            scores = distances * distances
            #print(scores.shape)
            # input()
            def proj_vlit_flit(vlit):
                if vlit < problem.n_vars:
                    return vlit - batch * n_vars_per_batch
                else:
                    return ((vlit - problem.n_vars) - batch * n_vars_per_batch) + n_vars_per_batch

            def decode_kmeans_A(vlit):
                # return scores[proj_vlit_flit(vlit), 0] + scores[proj_vlit_flit(flip_vlit(vlit)), 1] > \
                #     scores[proj_vlit_flit(vlit), 1] + scores[proj_vlit_flit(flip_vlit(vlit)), 0]
                return scores[proj_vlit_flit(vlit), 0] > scores[proj_vlit_flit(vlit), 1]

            decode_kmeans_B = (lambda vlit: not decode_kmeans_A(vlit))
            
            if self.solves(problem, batch, decode_kmeans_A): solutions.append(reify(decode_kmeans_A))
            elif self.solves(problem, batch, decode_kmeans_B): solutions.append(reify(decode_kmeans_B))
            else: solutions.append(None)

        return solutions
    def find_solutions(self, problem):
        def flip_vlit(vlit):
            if vlit < problem.n_vars: return vlit + problem.n_vars
            else: return vlit - problem.n_vars

        n_batches = len(problem.is_sat)
        n_vars_per_batch = problem.n_vars // n_batches

        d = self.build_feed_dict(problem)
        all_votes, final_lits, logits, costs = self.sess.run([self.all_votes, self.final_lits, self.logits, self.predict_costs], feed_dict=d)

        solutions = []
        for batch in range(len(problem.is_sat)):
            def reify(phi):
                xs = list(zip([phi(vlit) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)],
                              [phi(flip_vlit(vlit)) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)]))
                def one_of(a, b): return (a and (not b)) or (b and (not a))
                assert(all([one_of(x[0], x[1]) for x in xs]))
                return [x[0] for x in xs]
            L = np.reshape(final_lits, [2 * n_batches, n_vars_per_batch, self.opts.d])
            L1 = L[batch, :, :]
            kmeans = KMeans(n_clusters=2, random_state=0).fit(L1)
            distances = kmeans.transform(L1)
            scores = distances * distances
            answer = scores[:,0]>scores[:,1]
            answert = scores[:,0]<scores[:,1]
            print(answer)
            print(scores[:,0]-scores[:,1])
            # input()
            if self.solvesnew(problem, batch, answer, n_vars_per_batch):
                solutions.append(answer)
            elif self.solvesnew(problem, batch, answert, n_vars_per_batch):
                solutions.append(answert)
            else:
                solutions.append(answer)
        return solutions

    def solves(self, problem, batch, phi):
        start_cell = sum(problem.n_cells_per_batch[0:batch])
        end_cell = start_cell + problem.n_cells_per_batch[batch]

        if start_cell == end_cell:
            # no clauses
            return 1.0

        current_clause = problem.L_unpack_indices[start_cell, 1]
        current_clause_satisfied = False

        for cell in range(start_cell, end_cell):
            next_clause = problem.L_unpack_indices[cell, 1]

            # the current clause is over, so we can tell if it was unsatisfied
            if next_clause != current_clause:
                if not current_clause_satisfied:
                    return False

                current_clause = next_clause
                current_clause_satisfied = False

            if not current_clause_satisfied:
                vlit = problem.L_unpack_indices[cell, 0]
                
                if phi(vlit):
                    current_clause_satisfied = True

        # edge case: the very last clause has not been checked yet
        if not current_clause_satisfied: return False
        return True
    def solvesnew(self,problem, batch, answer,dd):
        start_cell = sum(problem.n_cells_per_batch[0:batch])
        end_cell = start_cell + problem.n_cells_per_batch[batch]
        #print(start_cell)
        #print(end_cell)
        #input()
        if start_cell == end_cell:
            # no clauses
            return 1.0
        current_clause = problem.L_unpack_indices[start_cell, 1]
        current_clause_satisfied = False
        l = []
        i=0
        ii=0
        for cell in range(start_cell, end_cell):
            #print(cell)
            next_clause = problem.L_unpack_indices[cell, 1]
            # the current clause is over, so we can tell if it was unsatisfied
            if next_clause != current_clause:
                if not current_clause_satisfied:
                    #print("%d--%d"%(current_clause,batch))
                    return False
                current_clause = next_clause
                current_clause_satisfied = False
                l.append(ii/i)
                i = 0
                ii = 0
  
            if not current_clause_satisfied:
                i+=1
                vlit = problem.L_unpack_indices[cell, 0]
                #print("-------------%d,%d"%(vlit,problem.n_vars))
                if vlit >= problem.n_vars:
                    vlit -= problem.n_vars+dd*batch
                    #print(vlit)
                    if answer[vlit] == False:
                        #print("c%d"%current_clause)
                        ii+=1
                        current_clause_satisfied = True
                else:
                    vlit -= dd*batch
                    #print(vlit)
                    if answer[vlit] == True:
                        ii+=1
                        #print("c%d"%current_clause)
                        current_clause_satisfied = True
        # edge case: the very last clause has not been checked yet
        if not current_clause_satisfied:
            #print("%d--%d"%(current_clause,batch))
            return False
        #print("%d--%d"%(current_clause,batch))
        #print(l)
        return True

    def find_psolutions(self, problem):
        def flip_vlit(vlit):
            if vlit < problem.n_vars: return vlit + problem.n_vars
            else: return vlit - problem.n_vars
        def distance(a,b):
            add = 0
            for i,_ in enumerate(a):
                add+=(a[i]-b[i])**2
            return add
        n_batches = len(problem.is_sat)
        n_vars_per_batch = problem.n_vars // n_batches

        d = self.build_feed_dict(problem)
        all_votes, final_lits, logits, costs = self.sess.run([self.all_votes, self.final_lits, self.logits, self.predict_costs], feed_dict=d)

        solutions = []
        for batch in range(len(problem.is_sat)):
            def reify(phi):
                xs = list(zip([phi(vlit) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)],
                              [phi(flip_vlit(vlit)) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)]))
                def one_of(a, b): return (a and (not b)) or (b and (not a))
                assert(all([one_of(x[0], x[1]) for x in xs]))
                return [x[0] for x in xs]
            L = np.reshape(final_lits, [2 * n_batches, n_vars_per_batch, self.opts.d])
            L1 = L[batch, :, :]
            L2 = np.concatenate([L[batch, :, :], L[n_batches + batch, :, :]], axis=0)
            '''print(distance(L2[-1],L2[-2]))
            input()
            kmeans = KMeans(n_clusters=2, random_state=0).fit(L2)
            distances = kmeans.transform(L2)
            scores = distances * distances
            answer = scores[:,0]/(scores[:,0]+scores[:,1])'''
            return L2

