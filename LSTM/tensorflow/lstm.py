import tensorflow as tf;  
import numpy as np;

class LSTM(object):

        
    def build_input(self):
        config = self.config
        x  = tf.placeholder(tf.float32, shape=(None, config.embedding_dim), name='x')
        y1 = tf.placeholder(tf.int32, shape=(None,), name='y1')
        y2 = tf.placeholder(tf.int32, shape=(None,), name='y2')
        return x, y1, y2
     
    
    def build_loss(self, logits_a, logits_p):
        logits = tf.concat([logits_a, logits_p], 0)
        # logits = tf.clip_by_value(logits, 1e-10, 0.999)
        y = tf.concat([self.y1, self.y2], 0)
        # logits = tf.clip_by_value(logits_a, 1e-10, 0.999)
        # y = self.y1
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        # y = tf.one_hot(y, 3)
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_sum(loss)
        return loss
    
    
    def build_optimizer(self, loss):
        config = self.config
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        return train_op
    
    
    def evaluate(self, out, y):
        correct = tf.equal(tf.argmax(out, axis=1), tf.to_int64(y))
        correct = tf.reduce_mean(tf.cast(correct, tf.float32))
        correct = tf.equal(correct, 1)
        return correct
        
        
    def __init__(self, config):
        self.config = config
        self.init_state = []
        self.final_state = []

        self.x, self.y1, self.y2 = self.build_input()
        
        for i in ['a','p']:
            with tf.variable_scope("rnn_"+i):
                with tf.variable_scope("gru_cell"):
                    cell = tf.nn.rnn_cell.BasicLSTMCell(config.gru_hidden_size)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.drop_rate)
                    init_state = cell.zero_state(1, tf.float32)
                self.init_state.append(init_state)
                r, final_state = tf.nn.dynamic_rnn(cell, tf.reshape(self.x, [1, -1, config.embedding_dim]), initial_state=init_state)
                self.final_state.append(final_state)
                r = tf.reshape(r, [-1, config.gru_hidden_size])
            if i == 'a':
                r_a = r
            else:
                r_p = r
        
        with tf.variable_scope("dense_out_a"):
            C_a = tf.Variable(tf.random_normal([config.gru_hidden_size, 3]), name='C')
            logits_a = tf.matmul(r_a, C_a)
            out_a = tf.nn.softmax(logits_a)
            
        with tf.variable_scope("dense_out_p"):
            C_p = tf.Variable(tf.random_normal([config.gru_hidden_size, 3]), name='C')
            logits_p = tf.matmul(r_p, C_p)
            out_p = tf.nn.softmax(logits_p)
        
        with tf.variable_scope("loss"):
            loss = self.build_loss(logits_a, logits_p)
            self.loss = loss
        with tf.variable_scope("optimizer"):
            optimizer = self.build_optimizer(loss)
            self.optimizer = optimizer
            
        with tf.variable_scope("evaluate"):
            self.correct_a = self.evaluate(out_a, self.y1)
            self.correct_p = self.evaluate(out_p, self.y2)
            
        self.out_a = tf.argmax(out_a, axis=1)
            
            
             
                  
    