import tensorflow as tf;  
import numpy as np;

class CMLA(object):

        
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
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(loss)
        # y = tf.one_hot(y, 3)
        # loss = -tf.reduce_sum(y * tf.log(logits))
        return loss
    
    
    def build_optimizer(self, loss):
        config = self.config
        '''
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        '''
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op
        
        
    def evaluate(self, out, y):
        correct = tf.equal(tf.argmax(out, axis=1), tf.to_int64(y))
        correct = tf.reduce_mean(tf.cast(correct, tf.float32))
        correct = tf.equal(correct, 1)
        return correct
     
     
    def at_mul(self, h, g, u):
        config = self.config
        '''
        h = tf.tile(h, [1, config.attention_slice])
        h = tf.reshape(h, [-1, config.attention_slice, config.embedding_dim])
        hg = tf.multiply(h,g)
        u = tf.transpose(u)
        u = tf.tile(u, [tf.shape(hg)[0],1])
        u = tf.reshape(u, [tf.shape(hg)[0],config.embedding_dim,1])
        hgu = tf.matmul(hg, u)
        hgu = tf.reshape(hgu, [tf.shape(hg)[0], config.attention_slice])
        '''
        h = tf.tile(h, [config.attention_slice, 1])
        h = tf.reshape(h, [config.attention_slice,-1,config.embedding_dim])
        hg = tf.matmul(h,g)
        u = tf.transpose(u)
        u = tf.tile(u, [config.attention_slice,1])
        u = tf.reshape(u, [config.attention_slice,config.embedding_dim,1])
        hgu = tf.matmul(hg,u)
        hgu = tf.reshape(hgu, [config.attention_slice, -1])
        hgu = tf.transpose(hgu)
        # hgu = tf.matmul(h, tf.transpose(g))
        
        return hgu
    
    
    def f(self, h, G, D, u1, u2):
        config = self.config
        b1 = self.at_mul(h, G, u1)
        b2 = self.at_mul(h, D, u2)
        b = tf.tanh(tf.concat([b1, b2], 1))
        return b
    
    
    def single_layer(self, u_a, u_p):
        config = self.config
        x = self.x
        for i in ['a', 'p']:
            with tf.variable_scope("attention_"+i):
                
                G = tf.Variable(tf.random_normal([config.attention_slice, config.embedding_dim, config.embedding_dim]), name='G')
                D = tf.Variable(tf.random_normal([config.attention_slice, config.embedding_dim, config.embedding_dim]), name='D')
                b = self.f(x, G, D, u_a, u_p)
                
                with tf.variable_scope("rnn"):
                    with tf.variable_scope("gru_cell"):
                        cell = tf.nn.rnn_cell.GRUCell(config.gru_hidden_size)
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.drop_rate)
                        init_state = cell.zero_state(1, tf.float32)
                    self.gru_init_state.append(init_state)
                    r, final_state = tf.nn.dynamic_rnn(cell, tf.reshape(b, [1, -1, 2*config.attention_slice]), initial_state=init_state)
                    self.gru_final_state.append(final_state)
                r = tf.reshape(r, [-1, config.gru_hidden_size])
                v = tf.Variable(tf.random_normal([1, config.gru_hidden_size]), name='v')
                e = tf.matmul(v, tf.transpose(r))
            if i == 'a':
                r_a = r
                e_a = e
            else:
                r_p = r
                e_p = e
        return r_a, r_p, e_a, e_p
    
    
    def update_for_u(self, u_pre, V, e):
        a = tf.div(tf.exp(e), tf.reduce_sum(tf.exp(e)))
        o = tf.matmul(a, self.x)
        u = tf.add(tf.tanh(tf.matmul(u_pre, V)), o)
        return u
    
    
    def __init__(self, config):
        self.config = config
        self.gru_init_state = []
        self.gru_final_state = []

        self.x, self.y1, self.y2 = self.build_input()
        with tf.variable_scope("layer_0"):
            u_a = tf.Variable(tf.random_uniform([1, config.embedding_dim], minval=-0.2, maxval=0.2, dtype=tf.float32), name='u_a')
            u_p = tf.Variable(tf.random_uniform([1, config.embedding_dim], minval=-0.2, maxval=0.2, dtype=tf.float32), name='u_p')
            r_a_0, r_p_0, e_a_0, e_p_0 = self.single_layer(u_a, u_p)
        
        with tf.variable_scope("plusU"):
            with tf.variable_scope("attention_a"):
                V_a = tf.Variable(tf.random_normal([config.embedding_dim, config.embedding_dim]), name='V')
                u_a_plus = self.update_for_u(u_a, V_a, e_a_0)
            with tf.variable_scope("attention_p"):
                V_p = tf.Variable(tf.random_normal([config.embedding_dim, config.embedding_dim]), name='V')
                u_p_plus = self.update_for_u(u_p, V_p, e_p_0)
        
        with tf.variable_scope("layer_1"):
            r_a_1, r_p_1, e_a_1, e_p_1 = self.single_layer(u_a_plus, u_p_plus)
        
        with tf.variable_scope("dense_out_a"):
            C_a = tf.Variable(tf.random_normal([config.gru_hidden_size, 3]), name='C')
            logits_a = tf.matmul(tf.add(r_a_0, r_a_1), C_a)
            out_a = tf.nn.softmax(logits_a)
        with tf.variable_scope("dense_out_p"):
            C_p = tf.Variable(tf.random_normal([config.gru_hidden_size, 3]), name='C')
            logits_p = tf.matmul(tf.add(r_p_0, r_p_1), C_p)
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
            
        self.logits_a = logits_a
        self.out_a = tf.argmax(out_a, axis=1)
            
            
             
                  
    