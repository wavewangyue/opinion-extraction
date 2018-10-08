#coding:utf-8
import sys
import keras
import json
import numpy as np
import gensim
reload(sys)
sys.setdefaultencoding('utf8')

MAX_SEQUENCE_LENGTH = 15
EMBEDDING_DIM = 200
HIDDEN_SIZE = 80

all_words = {}
wtf_words = {} 

def load_doc_to_vecs(path):
    import string
    model = gensim.models.KeyedVectors.load_word2vec_format('yelp.vector.bin', binary=True)
    lines = open(path).read().split('\n')
    vecs = []
    for line in lines:
        line = line.translate(None, string.punctuation)
        words = line.split(' ')
        line_vecs = []
        for word in words:
            if word not in all_words:
                all_words[word] = ''
            if word in model:
                line_vecs.append(model[word])
            else:
                line_vecs.append(np.zeros(EMBEDDING_DIM))
                if word not in wtf_words:
                    wtf_words[word] = ''
        if len(line_vecs) > MAX_SEQUENCE_LENGTH:
            line_vecs = line_vecs[:MAX_SEQUENCE_LENGTH]
        else:
            line_vecs = [np.zeros(EMBEDDING_DIM)]*(MAX_SEQUENCE_LENGTH-len(line_vecs)) + line_vecs
        vecs.append(line_vecs)
    return np.array(vecs), lines


def load_label_to_vecs(path):
    lines = open(path).read().split('\n')
    vecs = []
    for line in lines:
        labels = []
        for label in line.split(' '):
            if label == '0':
                labels.append([1, 0, 0])
            elif label == '1':
                labels.append([0, 1, 0])
            else:
                labels.append([0, 0, 1])
        if len(labels) > MAX_SEQUENCE_LENGTH:
            labels = labels[:MAX_SEQUENCE_LENGTH]
        else:
            labels = [[0, 0, 0]]*(MAX_SEQUENCE_LENGTH-len(labels)) + labels
        vecs.append(labels)
    return np.array(vecs)
       
        

print '(1)load data...'
x_train, x_docs = load_doc_to_vecs('train_docs.txt')
y_train_a = load_label_to_vecs('train_labels_a.txt')
y_train_p = load_label_to_vecs('train_labels_p.txt')
print 'there are '+str(len(all_words))+' words totally'
print 'there are '+str(len(wtf_words))+' words not be embeded'
print 'train docs: '+str(x_train.shape)
print 'train labels of aspect: '+str(y_train_a.shape)
print 'train labels of opinion: '+str(y_train_p.shape)


print '(2)build model...'
from keras.layers import Dense, Activation, Reshape, Lambda, Input
from keras.layers import Embedding, GRU, LSTM
from keras.models import Model
from attention import MultiCoupledAttentionLayer1
from keras.utils import plot_model

main_input = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dtype='float32')
lstm1 = LSTM(HIDDEN_SIZE, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, name='lstm1')(main_input)
out1 = Dense(3, activation='softmax', name='out1')(lstm1)
model = Model(inputs=main_input, outputs=out1)
model.summary()


print '(3)run model...'
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.fit(x_train, y_train_a, epochs=5, batch_size=32)
# model.save('lstm_model.h5')