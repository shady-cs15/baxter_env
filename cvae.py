import tensorflow as tf
import numpy as np

class CVAE(object):
	def __init__(self, batch_size, activation=tf.nn.tanh, input_dim=20, encoder_hiddims=[50, 100], \
					sample_dim=200, decoder_hiddims=[100, 50, 20]):
		self.input = tf.placeholder(tf.float32,[batch_size, input_dim])
		self.output = tf.placeholder(tf.float32,[batch_size, decoder_hiddims[-1]])
		self.nlayers = 1 + len(encoder_hiddims) + 1 + len(decoder_hiddims)
		self.batch_size = batch_size
		self.activation = activation
		self.input_dim = input_dim
		self.enc_dims = encoder_hiddims
		self.sample_dim = sample_dim
		self.dec_dims = decoder_hiddims
		self.output_dim = decoder_hiddims[-1]
		self.layers = [tf.concat([self.input, self.output], axis=1)]

		assert self.input.shape[0] == batch_size
		assert self.input.shape[1] == self.input_dim
		assert self.output.shape[0] == batch_size
		assert self.output.shape[1] == self.output_dim

		self.loss = 0.
		self.mu = None
		self.log_sigma = None
		self.optimizer = None
	
	def kldiv_loss(self, mu, log_sigma):
		return tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(log_sigma) + mu**2.0 - 1. - log_sigma, axis=1))

	def recon_loss(self, preds, labels):
		return tf.reduce_mean(tf.reduce_sum((preds-labels)**2.0, axis=1))

	def sample(self, mu, log_sigma, batch_size, sample_dim):
		eps = tf.random_normal(shape=(batch_size, sample_dim))
		return mu + tf.exp(log_sigma / 2) * eps

	def build_model(self):
		# build encoder
		for i in range(len(self.enc_dims)):
			next_layer = tf.layers.dense(self.layers[-1], self.enc_dims[i], activation=self.activation)
			self.layers.append(next_layer)

		# build sampling layer
		self.mu = tf.layers.dense(self.layers[-1], self.sample_dim, activation=None)
		self.log_sigma = tf.layers.dense(self.layers[-1], self.sample_dim, activation=None)
		z = self.sample(self.mu, self.log_sigma, self.batch_size, self.sample_dim)
		self.layers.append(tf.concat([z, self.input], axis=1))

		# build decoder 
		for i in range(len(self.dec_dims)):
			next_layer = tf.layers.dense(self.layers[-1], self.dec_dims[i], activation=self.activation)
			self.layers.append(next_layer)

		assert self.nlayers == len(self.layers)

		self.loss = self.kldiv_loss(self.mu, self.log_sigma) + self.recon_loss(self.layers[-1], self.output)

	def build_optimizer(self, op='sgd', lr=0.01):
		if op=='sgd':
			self.optimizer = tf.train.GradientDescentOptimizer(lr)
		elif op=='adam':
			self.optimizer = tf.train.AdamOptimizer(lr)
		else:
			raise NotImplementedError
		self.optimizer = self.optimizer.minimize(self.loss)

class Trainer(object):
	def __init__(self, model, train_x, train_y, batch_size):
		assert train_x.shape[1] == model.input_dim
		assert train_y.shape[1] == model.input_dim
		assert train_x.shape[0] == train_y.shape[0]
		self.model = model
		self.train_x = train_x
		self.train_y = train_y
		self.batch_size = batch_size
		import os
		if not os.path.exists('./weights/'):
			os.makedirs('./weights/')
		self.saver = tf.train.Saver()

	def train(self, nepochs):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print 'starting training..'
			for epoch in range(nepochs):
				losses = []
				for batch_idx in range(self.train_x.shape[0]/self.batch_size):
					loss, _ = sess.run([self.model.loss, self.model.optimizer], \
						feed_dict={self.model.input: self.train_x[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size], \
									self.model.output: self.train_y[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]})
					losses.append(loss)
				losses = np.mean(loss)
				print 'epoch:', epoch+1, 'loss:', losses
				self.saver.save(sess, 'weights/model.ckpt')

'''
from cvae import CVAE, Trainer

model = CVAE(...)
model.build_model()
model.build_optimizer()

trainer = Trainer(model, train_x, train_y, batch_size)
trainer.train(nepochs)
'''
