import numpy as np
import tensorflow as tf

from sac import ValueNetwork, CriticNetwork, ActorNetwork

with tf.Session() as sess:
	vnet = ValueNetwork(sess, 2, 0.5, 0.8)
	vnet2 = ValueNetwork(sess, 2, 0.5, 0.8, scope='psi_')
	cnet = CriticNetwork(sess, 2, 2, vnet2, 0.9, 0.2)
	
	sess.run(tf.global_variables_initializer())
	for var in tf.trainable_variables():
		print var.name
	
	for i in range(5):
		p, l, _ = vnet.train(np.zeros([5, 2]), np.ones([5, 1]))
		print l
	p = vnet.predict(np.zeros([5, 2]))
	print p

	x = np.zeros([5, 2])
	y = np.ones([5, 1])
	print vnet2.predict(np.zeros([5, 2]))
	vnet2.update(vnet)
	print vnet2.predict(np.zeros([5, 2]))
	print cnet.value_net.predict(x)	

	for i in range(5):
		p, l, _ = cnet.train(x, x, x, y, np.zeros([5, 1]))
		print l	
