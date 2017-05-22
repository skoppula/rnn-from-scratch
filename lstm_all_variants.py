import tensorflow as tf
import numpy as np

import utils
import data

import random
import argparse
import sys
import os
from data import *

BATCH_SIZE = 256
DATASET = None
paths = {'sms':['data/sms/','data/sms/sms.txt'],
            'shakespeare':['data/shakespeare/','data/shakespeare/shakespeare.txt'],
            'paulg':['data/paulg/','data/paulg/paulg.txt']}


class LSTM_rnn():

    def __init__(self, state_size, num_classes, dataset, variant, model_name):
        self.state_size = state_size
        self.num_classes = num_classes
        self.ckpt_path = 'ckpt/'+ dataset + '/'
        self.model_name = model_name
        print("Model name", model_name)

        def __graph__():
            tf.reset_default_graph()

            # inputs
            xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
            ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
            
            # embeddings
            embs = tf.get_variable('emb', [num_classes, state_size])
            rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
            
            # initial hidden state
            init_state = tf.placeholder(shape=[2, None, state_size], dtype=tf.float32, name='initial_state')

            # initializer
            xav_init = tf.contrib.layers.xavier_initializer

            # params
            W = tf.get_variable('W', shape=[4, self.state_size, self.state_size], initializer=xav_init())
            U = tf.get_variable('U', shape=[4, self.state_size, self.state_size], initializer=xav_init())
            # b = tf.get_variable('b', shape=[self.state_size], initializer=tf.constant_initializer(0.))

            def sigmoid_approx(tensor):
                return tf.maximum(0.0,tf.minimum(1.0, 0.125*tensor + 0.5))

            def step_additive_no_sigmoid_linear(prev, x):
                previous_output, previous_state = tf.unstack(prev)

                # i = tf.sigmoid(tf.matmul(x,U[0]) + tf.matmul(previous_output,W[0]))
                f = sigmoid_approx(tf.matmul(x,U[1]) + tf.matmul(previous_output,W[1]))
                i = 1-f
                new_state_contrib = sigmoid_approx(tf.matmul(x,U[3]) + tf.matmul(previous_output,W[3]))

                new_state = tf.nn.relu(previous_state - f) + tf.nn.relu(new_state_contrib - i)

                out_gate = sigmoid_approx(tf.matmul(x,U[2]) + tf.matmul(previous_output,W[2]))
                output = sigmoid_approx(new_state)/6.0 - out_gate

                return tf.stack([output, new_state])

            def step_additive_sigmoid(prev, x):
                previous_output, previous_state = tf.unstack(prev)

                # i = tf.sigmoid(tf.matmul(x,U[0]) + tf.matmul(previous_output,W[0]))
                # multiply by max of previous output?
                f = tf.sigmoid(tf.matmul(x,U[1]) + tf.matmul(previous_output,W[1])) 
                i = 1-f
                new_state_contrib = tf.sigmoid(tf.matmul(x,U[3]) + tf.matmul(previous_output,W[3]))

                new_state = tf.nn.relu(previous_state - f) + tf.nn.relu(new_state_contrib - i)

                out_gate = tf.sigmoid(tf.matmul(x,U[2]) + tf.matmul(previous_output,W[2]))
                output = tf.sigmoid(new_state) - out_gate

                return tf.stack([output, new_state])

            def step_normal(prev, x):
                previous_output, previous_state = tf.unstack(prev)

                i = tf.sigmoid(tf.matmul(x,U[0]) + tf.matmul(previous_output,W[0]))
                f = tf.sigmoid(tf.matmul(x,U[1]) + tf.matmul(previous_output,W[1]))
                new_state_contrib = tf.tanh(tf.matmul(x,U[3]) + tf.matmul(previous_output,W[3]))

                new_state = previous_state*f + new_state_contrib*i

                out_gate = tf.sigmoid(tf.matmul(x,U[2]) + tf.matmul(previous_output,W[2]))
                output = tf.tanh(new_state)*out_gate

                return tf.stack([output, new_state])

            def step_additive_no_sigmoid_relu6(prev, x):
                previous_output, previous_state = tf.unstack(prev)

                # i = tf.sigmoid(tf.matmul(x,U[0]) + tf.matmul(previous_output,W[0]))
                f = tf.nn.relu6(tf.matmul(x,U[1]) + tf.matmul(previous_output,W[1]))/6.0
                i = 1-f
                new_state_contrib = tf.nn.relu6(tf.matmul(x,U[3]) + tf.matmul(previous_output,W[3]))/6.0

                new_state = tf.nn.relu(previous_state - f) + tf.nn.relu(new_state_contrib - i)

                out_gate = tf.nn.relu6(tf.matmul(x,U[2]) + tf.matmul(previous_output,W[2]))/6.0
                output = tf.nn.relu6(new_state)/6.0 - out_gate

                return tf.stack([output, new_state])

            step = None
            if variant == 'additive_no_sigmoid_relu6':
                step = step_additive_no_sigmoid_relu6
            elif variant == 'normal':
                step = step_normal
            elif variant == 'additive_sigmoid':
                step = step_additive_sigmoid
            elif variant == 'additive_no_sigmoid_linear':
                step = step_additive_no_sigmoid_linear
            else:
                print("Variant not recognized!")
                sys.exit(1)

            states = tf.scan(step,
                    tf.transpose(rnn_inputs, [1,0,2]),
                    initializer=init_state)
            
            # predictions
            V = tf.get_variable('V', shape=[state_size, num_classes],
                                initializer=xav_init())
            bo = tf.get_variable('bo', shape=[num_classes],
                                 initializer=tf.constant_initializer(0.))

            # get last state before reshape/transpose
            last_state = states[-1]

            states = tf.transpose(states, [1,2,0,3])[0]
            # st_shp = tf.shape(states)

            # flatten states to 2d matrix for matmult with V
            # states_reshaped = tf.reshape(states, [st_shp[0] * st_shp[1], st_shp[2]])
            states_reshaped = tf.reshape(states, [-1, state_size])
            logits = tf.matmul(states_reshaped, V) + bo

            predictions = tf.nn.softmax(logits)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ys_)
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
            
            self.xs_ = xs_
            self.ys_ = ys_
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.last_state = last_state
            self.init_state = init_state

        print('building graph...')
        __graph__()


    def train(self, train_set, val_set, num_steps_trn_epoch, num_steps_val_epoch, epochs=600):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            try:
                for i in range(epochs):
                    train_loss = 0
                    for j in range(num_steps_trn_epoch):
                        xs, ys = next(train_set)
                        batch_size = xs.shape[0]
                        _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict = {
                                self.xs_ : xs,
                                self.ys_ : ys.flatten(),
                                self.init_state : np.zeros([2, batch_size, self.state_size])
                            })

                        train_loss += train_loss_

                    val_loss = 0
                    for j in range(num_steps_val_epoch):
                        xs, ys = next(val_set)
                        batch_size = xs.shape[0]
                        val_loss_ = sess.run(self.loss, feed_dict = {
                                self.xs_ : xs,
                                self.ys_ : ys.flatten(),
                                self.init_state : np.zeros([2, batch_size, self.state_size])
                            })
                        val_loss += val_loss_

                    if i % 100 == 0:
                        saver = tf.train.Saver()
                        saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
                    print("epoch", i, "train_loss:", train_loss/100.0, "val_loss:", val_loss/100.0)


            except KeyboardInterrupt:
                print('interrupted by user at ' + str(i))

            

    def generate(self, idx2w, w2idx, num_words=100, separator=' '):

        random_init_word = random.choice(idx2w)
        current_word = w2idx[random_init_word]

        with tf.Session() as sess:
            # init session
            sess.run(tf.global_variables_initializer())

            # restore session
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            words = [current_word]
            state = None
            for i in range(num_words):
                if state:
                    feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                            self.init_state : state_}
                else:
                    feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                            self.init_state : np.zeros([2, 1, self.state_size])}

                preds, state_ = sess.run([self.predictions, self.last_state], feed_dict=feed_dict)

                state = True
                current_word = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
                words.append(current_word)

        return separator.join([idx2w[w] for w in words])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Long Short Term Memory RNN for Text Hallucination, built with tf.scan')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate', action='store_true',
                        help='generate text')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    parser.add_argument('-n', '--num_words', required=False, type=int,
                        help='number of words to generate')
    parser.add_argument('-d', '--dataset', required=True,
                        help='dataset to use: paulg, shakespeare, sms')
    parser.add_argument('-v', '--variant', required=True,
                        help='lstm variant to use: additive_no_sigmoid_relu6,normal,additive_sigmoid,additive_no_sigmoid_linear')

    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':

    args = parse_args()
    DATASET = args['dataset']
    process_data(paths[DATASET][0], paths[DATASET][1])
    print(args)
    X_trn, y_trn, X_test, y_test, idx2w, w2idx = data.load_data('data/' + DATASET + '/')
    print("fetched data. trn/test data shape: ", X_trn.shape, y_trn.shape, X_test.shape, y_test.shape)
    
    num_steps_trn_epoch = X_trn.shape[0]/BATCH_SIZE
    num_steps_val_epoch = X_test.shape[0]/BATCH_SIZE
    print("num steps in trn and val epochs", num_steps_trn_epoch, num_steps_val_epoch)

    model = LSTM_rnn(state_size = 512, num_classes=len(idx2w), dataset=DATASET, variant=args['variant'], model_name=args['variant'])
    print("created model.")

    if args['train']:
        train_set = utils.rand_batch_gen(X_trn, y_trn, batch_size=BATCH_SIZE)
        val_set = utils.rand_batch_gen(X_test, y_test, batch_size=BATCH_SIZE)
        print("starting to train model!")
        model.train(train_set, val_set, num_steps_trn_epoch, num_steps_val_epoch)

    elif args['generate']:
        text = model.generate(idx2w, w2idx,
                num_words=args['num_words'] if args['num_words'] else 100,
                separator='')

        print('______Generated Text_______')
        print(text)
        print('___________________________')
