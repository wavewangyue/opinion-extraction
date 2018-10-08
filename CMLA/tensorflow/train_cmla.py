import gensim
import numpy as np
import tensorflow as tf
import time
import random
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
all_words = {}
wtf_words = {}   


class Config(object):
    embedding_dim = 200
    attention_slice = 15
    gru_hidden_size = 30
    batch_size = 1
    num_layer = 2
    learning_rate = 0.0007
    drop_rate = 0.5
    max_grad_norm = 5
    
    
def load_doc_to_vecs(path, config):
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
                line_vecs.append(np.zeros(config.embedding_dim))
                if word not in wtf_words:
                    wtf_words[word] = ''
        vecs.append(line_vecs)
    return vecs, lines


def load_label_to_vecs(path):
    lines = open(path).read().split('\n')
    vecs = []
    for line in lines:
        labels = line.split(' ')
        vecs.append(labels)
    return vecs


if __name__ == '__main__':
    config = Config()
    
    print ''
    print '(1) load data and trans to vecs...' 
    x_train, x_docs = load_doc_to_vecs('train_docs.txt', config)
    y_train_a = load_label_to_vecs('train_labels_a.txt')
    y_train_p = load_label_to_vecs('train_labels_p.txt')
    x_train = np.array(x_train)
    x_test, x_docs_test = load_doc_to_vecs('test_docs.txt', config)
    y_test_a = load_label_to_vecs('test_labels_a.txt')
    y_test_p = load_label_to_vecs('test_labels_p.txt')
    x_test = np.array(x_test)
    print 'there are '+str(len(all_words))+' words totally'
    print 'there are '+str(len(wtf_words))+' words not be embeded'
    print 'train docs: '+str(len(x_train))
    print 'train labels of aspect: '+str(len(y_train_a))
    print 'train labels of opinion: '+str(len(y_train_p))
    print 'test docs: '+str(len(x_test))
    print 'test labels of aspect: '+str(len(y_test_a))
    print 'test labels of opinion: '+str(len(y_test_p))
        
    print ''
    print '(2) build model...'
    from cmlapp import CMLA
    model = CMLA(config=config)
    
    print ''
    print '(3) train model...'
    epochs = 100
    with tf.Session() as sess:
        # merged = tf.summary.merge_all()
        tf.summary.FileWriter('graph', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        start = time.time()
        new_state = sess.run(model.gru_init_state)
        statistic_step = 3041
        total_loss = 0
        for e in range(epochs):    
            for i in range(len(x_train)):
                feed_dict = {model.x: x_train[i],
                             model.y1: y_train_a[i],
                             model.y2: y_train_p[i]}
                for ii, dd in zip(model.gru_init_state, new_state):
                    feed_dict[ii] = dd
                loss, new_state, _ = sess.run([model.loss, model.gru_final_state, model.optimizer], feed_dict=feed_dict)
                total_loss += loss
                end = time.time()
                if i == 0:
                    print '********************************************' 
                    print 'epoch: '+str(e)+' / '+str(epochs)
                    print 'steps: '+str(i)
                    print 'cost_time: '+str(end-start)
                    print 'loss: '+str(total_loss/statistic_step)
                    total_loss = 0
                
                if i == 0:
                    correct_a_num = 0
                    correct_p_num = 0
                    test_batch_size = len(x_train)
                    for j in range(test_batch_size):
                        # index = random.randint(0, len(x_train)-1)
                        feed_dict[model.x] = x_train[j]
                        feed_dict[model.y1] = y_train_a[j]
                        feed_dict[model.y2] = y_train_p[j]
                        correct_a, correct_p = sess.run([model.correct_a, model.correct_p], feed_dict=feed_dict)
                        if correct_a:
                            correct_a_num += 1
                        if correct_p:
                            correct_p_num += 1
                    score1 = float(correct_a_num)*100/test_batch_size
                    score2 = float(correct_p_num)*100/test_batch_size
                    print 'precision train: '+str(score1)+' '+str(score2)
                
                if i == 0:
                    correct_a_num = 0
                    correct_p_num = 0
                    test_batch_size = len(x_test)
                    for j in range(test_batch_size):
                        # index = random.randint(0, len(x_train)-1)
                        feed_dict[model.x] = x_test[j]
                        feed_dict[model.y1] = y_test_a[j]
                        feed_dict[model.y2] = y_test_p[j]
                        correct_a, correct_p = sess.run([model.correct_a, model.correct_p], feed_dict=feed_dict)
                        if correct_a:
                            correct_a_num += 1
                        if correct_p:
                            correct_p_num += 1
                    score1 = float(correct_a_num)*100/test_batch_size
                    score2 = float(correct_p_num)*100/test_batch_size
                    print 'precision test: '+str(score1)+' '+str(score2)
            
            
            
            
            
