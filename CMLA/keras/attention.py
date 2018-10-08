from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import GRU
import numpy as np

# Multi-Coupled Attention Layer
# Layer 1
class MultiCoupledAttentionLayer1(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MultiCoupledAttentionLayer1, self).__init__(**kwargs)

    def build(self, input_shape):
        self.G_a = self.add_weight(name='G_a', shape=(self.output_dim, input_shape[2], input_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        self.D_a = self.add_weight(name='D_a', shape=(self.output_dim, input_shape[2], input_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        self.G_p = self.add_weight(name='G_p', shape=(self.output_dim, input_shape[2], input_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        self.D_p = self.add_weight(name='D_p', shape=(self.output_dim, input_shape[2], input_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        self.u_a = self.add_weight(name='u_a', shape=(input_shape[2], 1),
                                      initializer='uniform',
                                      trainable=True)
        self.u_p = self.add_weight(name='u_p', shape=(input_shape[2], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(MultiCoupledAttentionLayer1, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        b_a_a = K.dot(x, self.G_a)
        b_a_a = K.dot(b_a_a, self.u_a)
        b_a_p = K.dot(x, self.D_a)
        b_a_p = K.dot(b_a_p, self.u_p) 
        b_a = K.concatenate([b_a_a,b_a_p],axis=3)
        
        
        b_p_p = K.dot(x, self.G_p)
        b_p_p = K.dot(b_p_p, self.u_p)
        b_p_a = K.dot(x, self.D_p)
        b_p_a = K.dot(b_p_a, self.u_a)
        b_p = K.concatenate([b_p_p,b_p_a],axis=3) 
        
        b = K.concatenate([b_a, b_p],axis=2)
        b = K.tanh(b)
        return b

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim*2, 2)

