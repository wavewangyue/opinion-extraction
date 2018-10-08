#coding:utf-8
import sys
import keras
import json
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')

MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 200
TEST_SPLIT = 0.2
ATTENTION_SLICES = 20
ATTENTION_LAYERS = 2

def label_to_vec(all_lines):
    vecs_form = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
    all_vecs = []
    for line in all_lines:
        labels = line.split(' ')
        line_vecs = []
        for label in labels:
            line_vecs.append(vecs_form[int(label)])
        if len(line_vecs) > MAX_SEQUENCE_LENGTH:
            line_vecs = line_vecs[:MAX_SEQUENCE_LENGTH]
        else:
            line_vecs = [[0,0,0]]*(MAX_SEQUENCE_LENGTH-len(line_vecs)) + line_vecs
        all_vecs.append(line_vecs)
    all_vecs = np.array(all_vecs)
    return all_vecs
       
        

print '(1)load data...'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train_docs = open('train_docs.txt').read().split('\n')
train_labels_a = open('train_labels_a.txt').read().split('\n')
train_labels_p = open('train_labels_p.txt').read().split('\n')
test_docs = open('test_docs.txt').read().split('\n')
test_labels_a = open('test_labels_a.txt').read().split('\n')
test_labels_p = open('test_labels_p.txt').read().split('\n')
all_docs = train_docs + test_docs

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_docs)
word_index = tokenizer.word_index
print 'word index size : '+str(len(word_index))
sequences = tokenizer.texts_to_sequences(all_docs)
all_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
train_data = all_data[:len(train_docs)]
test_data = all_data[len(train_docs):]
print('Shape of train data tensor:', train_data.shape)
print('Shape of test data tensor:', test_data.shape)

train_labels_a = label_to_vec(train_labels_a)
train_labels_p = label_to_vec(train_labels_p)
test_labels_a = label_to_vec(test_labels_a)
test_labels_p = label_to_vec(test_labels_p)
print('Shape of train aspect label tensor:', train_labels_a.shape)
print('Shape of train opinion tensor:', train_labels_p.shape)
print('Shape of test aspect tensor:', test_labels_a.shape)
print('Shape of test opinion tensor:', test_labels_p.shape)


print '(2) load word2vec as embedding...'
import gensim
from keras.utils import plot_model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('yelp.vector.bin', binary=True)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
not_in_model = 0
in_model = 0
for word, i in word_index.items(): 
    if word in w2v_model:
        in_model += 1
        embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')
    else:
        not_in_model += 1
print str(not_in_model)+' words not in w2v model'
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)



print '(3)build model...'
from keras.layers import Dense, Activation, Reshape, Lambda, Input
from keras.layers import Embedding, GRU
from keras.models import Model
from attention import MultiCoupledAttentionLayer1
from keras.utils import plot_model

main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
x = embedding_layer(main_input)
x = MultiCoupledAttentionLayer1(ATTENTION_SLICES)(x)
lamb1 = Lambda(lambda x: x[:,:,:,0], name='lambda1')(x)
lamb2 = Lambda(lambda x: x[:,:,:,1], name='lambda2')(x)
gru1 = GRU(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name='gru1')(lamb1)
gru2 = GRU(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name='gru2')(lamb2)
out1 = Dense(3, activation='softmax', name='out1')(gru1)
out2 = Dense(3, activation='softmax', name='out2')(gru2)
#e1 = Dense(1, name='e1')(gru1)
#e2 = Dense(1, name='e2')(gru2)
model = Model(inputs=main_input, outputs=[out1, out2])
model.summary()
plot_model(model, to_file='model.png',show_shapes=True)



print '(4)run model...'
from keras import optimizers
rmsprop = optimizers.RMSprop(lr=0.07)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['acc'])
model.fit(train_data, [train_labels_a, train_labels_p], epochs=10, batch_size=128)
model.save('cmla_model.h5')

print model.evaluate(test_data, [test_labels_a, test_labels_a])