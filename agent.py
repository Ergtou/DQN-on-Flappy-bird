from functools import reduce
import tensorflow as tf
from ops import linear, conv2d, clipped_error
from replay_memory import *
from utils import *

import time

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.action_size = 2
        self.screen_width = self.env.screen_width
        self.screen_height = self.env.screen_height
        self.history_length = 4
        self.memory_size = 50000
        self.memory = Memory(self.memory_size, self.screen_height, self.screen_width, self.history_length)
        self.e_greedy = 1.0
        self.reward_decay = 0.99
        #self.learning_rate = 1e-6
        self.learning_rate = 0.00025
        self.cnn_format = 'NHWC'
        self.times = 0

        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        self.w = {}
        self.t_w = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32', [None, self.screen_height, self.screen_width, self.history_length],
                                      name='s_t')
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t, 32, [8, 8], [4, 4], initializer, activation_fn,
                                                             self.cnn_format, name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1, 64, [4, 4], [2, 2], initializer, activation_fn,
                                                             self.cnn_format, name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2, 64, [3, 3], [1, 1], initializer, activation_fn,
                                                             self.cnn_format, name='l3')

            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')
            self.q_action = tf.argmax(self.q, axis=1)

        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32',
                                             [None, self.screen_height, self.screen_width, self.history_length],
                                             name='target_s_t')

            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 32, [8, 8], [4, 4],
                                                                        initializer, activation_fn, self.cnn_format,
                                                                        name='target_l1')
            self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1, 64, [4, 4], [2, 2], initializer,
                                                                        activation_fn, self.cnn_format,
                                                                        name='target_l2')
            self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2, 64, [3, 3], [1, 1], initializer,
                                                                        activation_fn, self.cnn_format,
                                                                        name='target_l3')

            shape = self.target_l3.get_shape().as_list()

            self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                linear(self.target_l4, self.env.action_size, name='target_q')

            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted

            self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')
            #self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def q_learning_mini_batch(self):
        assert self.memory.count > 4
        s_t, action, reward, terminal, s_t_plus_1 = self.memory.sample()
        # A batch of memory
        q_t_plus_1 = self.sess.run(self.target_q, {self.target_s_t: s_t_plus_1})

        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        target_q_t = (1. - terminal) * self.reward_decay * max_q_t_plus_1 + reward

        _, loss = self.sess.run([self.optim, self.loss], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
        })
        print(loss)

    def choose_action(self, observation):
        if np.random.uniform() < self.e_greedy:
            action = np.random.randint(0, self.action_size)
        else:
            with self.sess.as_default():
                action = self.q_action.eval({self.s_t: [observation]})[0]
        return action

    def play(self):
        Reach_terminal = False
        screens, reward, action, Reach_terminal = self.env.reset()
        while not Reach_terminal:
            action = self.choose_action(screens)
            new_screens, reward, action, Reach_terminal = self.env.step(action)
            self.memory.add(screens, reward, action, Reach_terminal)
            screens=new_screens
            self.times += 1

    def update_target_q_network(self):
        with self.sess.as_default():
            for name in self.w.keys():
                self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def save_weight(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './model.ckpt')
        del saver

    def restore_weight(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './model.ckpt')
        del saver
