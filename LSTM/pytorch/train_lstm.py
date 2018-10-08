# -*- coding:utf-8 -*-  
import torch
import torch.nn as nn	 
import torch.optim as optim 
import time
import gensim
import numpy as np
from torch.autograd import Variable 
import random
all_words = {}
wtf_words = {} 


class Config(object):
	embedding_dim = 200
	hidden_dim = 80
	batch_size = 1
	label_dim = 3
	learning_rate = 0.1
	drop_rate = 0.5

	
def load_data_to_vecs(path, style):
	if style=='x':
		model = gensim.models.KeyedVectors.load_word2vec_format('yelp.vector.bin', binary=True)
		lines = open(path).read().split('\n')
		vecs = []
		import string
		for line in lines:
			line = line.translate(None, string.punctuation)
			words = line.split(' ')
			line_vecs = []
			for word in words:
				if word not in all_words:
					all_words[word] = ''
				if word in model:
					line_vecs.append(model[word].tolist())
				else:
					line_vecs.append([0]*200)
					if word not in wtf_words:
						wtf_words[word] = ''
			vecs.append(line_vecs)
		return vecs
	if style=='y':
		lines = open(path).read().split('\n')
		vecs = []
		for line in lines:
			labels = line.split(' ')
			line_vecs = [int(label) for label in labels]
			vecs.append(line_vecs)
		return vecs
	
	
if __name__ == '__main__':
	config = Config()
	
	print ''
	print '(1) load data and trans to vecs...' 
	x_train = load_data_to_vecs('train_docs.txt', style='x')
	y_train = load_data_to_vecs('train_labels_a.txt', style='y')
	print 'there are '+str(len(all_words))+' words totally'
	print 'there are '+str(len(wtf_words))+' words not be embeded'
	print 'train docs: '+str(len(x_train))
	print 'train labels of aspect: '+str(len(y_train))
	
	print ''
	print '(2) build model...'
	from lstm import LSTMTagger
	model = LSTMTagger(config)
	loss_function = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
	print model
	
	print ''
	print '(3) train model...'
	epochs = 100
	start = time.time()
	for epoch in range(epochs):
		for i in range(len(x_train)):
			model.zero_grad()  # 清除网络先前的梯度值，梯度值是Pytorch的变量才有的数据，Pytorch张量没有
			model.hidden = model.init_hidden()  # 重新初始化隐藏层数据，避免受之前运行代码的干扰
			out = model(Variable(torch.FloatTensor(x_train[i])))
			loss = loss_function(out, Variable(torch.LongTensor(y_train[i])))
			loss.backward()
			optimizer.step()
			if i % 10 == 0:
				print '********************************************' 
				print 'epoch: '+str(epoch)+' / '+str(epochs)
				print 'steps: '+str(i)
				print 'cost_time: '+str(time.time()-start)
				print 'loss: '+str(loss.data.numpy()[0])
				hits = 0
			if i % 100 == 0:
				for j in range(128):
					index = random.randint(0, len(x_train)-1)
					out = model(Variable(torch.FloatTensor(x_train[j])))
					out = out.data.numpy()
					if all(np.argmax(out[index])==y_train[j][index] for index in range(len(y_train[j]))):
						hits += 1
				print 'precision: '+str(float(hits)/128)