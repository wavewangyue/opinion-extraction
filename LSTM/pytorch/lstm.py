import torch
from torch.autograd import Variable 
import torch.nn as nn	
import torch.nn.functional as F		
	  

class LSTMTagger(nn.Module):

	def __init__(self, config):
		super(LSTMTagger, self).__init__()
		self.embedding_dim = config.embedding_dim
		self.hidden_dim = config.hidden_dim
		self.label_dim = config.label_dim
		
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
		self.dense = nn.Linear(self.hidden_dim, self.label_dim)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		return (Variable(torch.zeros(1, 1, self.hidden_dim)),
				Variable(torch.zeros(1, 1, self.hidden_dim)))

	def forward(self, sentence):
		lstm_out, self.hidden = self.lstm(sentence.view(-1, 1, self.embedding_dim), self.hidden)
		logits = self.dense(lstm_out.view(-1,self.hidden_dim))
		out = F.log_softmax(logits)
		return out
	
		
	
