import tensorflow as tf

class ValueNetwork(object):
	def __init__(self):
		pass

	def _build_network(self, scope='Value', reuse=False):
		state = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		with tf.variable_scope(scope+'/dense1'):
			net = tf.layers.dense(state, 400, activation=tf.nn.relu, reuse=reuse)
			net = tf.contrib.layers.layer_norm(net)
		with tf.variable_scope(scope+'/dense2'):
			net = tf.layers.dense(net, 300, activation=tf.nn.relu, reuse=reuse)
			net = tf.contrib.layers.layer_norm(net)
		with tf.variable_scope(scope+'/dense3'):
			net = tf.layers.dense(net, 200, activation=tf.nn.relu, reuse=reuse)
			net = tf.contrib.layers.layer_norm(net)
		with tf.variable_scope(scope+'/dense4'):
			preds = tf.layers.dense(net, 1, activation=None, reuse=reuse)
		return state, preds

	def _build_optimizer(self):
		self.inputs, self.preds = self._build_network()
		self.net_params = tf.trainable_variables(scope='Value')

	def predict(self, state):
		pass

	def train(self):
		pass

class CriticNetwork(object):
	# value_net is a ValueNetwork object
	def __init__(self, sess, state_dim, action_dim, value_net, gamma, lambda_q):
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.value_net = value_net
		self.gamma = gamma
		self.lambda_q = lambda_q

		self._build_optimizer()

	def _build_network(self, scope='Critic', reuse=False):
		state = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		action = tf.placeholder(tf.float32, shape=[None, self.action_dim])
		next_state = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		reward = tf.placeholder(tf.float32, shape=[None, 1])
		terminal = tf.placeholder(tf.float32, shape=[None, 1])
		with tf.variable_scope(scope+'/dense1'):
			net = tf.layers.dense(state, 400, activation=tf.nn.relu, reuse=reuse)
			net = tf.contrib.layers.layer_norm(net)
		with tf.variable_scope(scope+'/dense2'):
			net = tf.layers.dense(tf.concat([net, action], 1), 300, activation=tf.nn.relu, reuse=reuse)
			net = tf.contrib.layers.layer_norm(net)
		with tf.variable_scope(scope+'/dense3'):
			net = tf.layers.dense(tf.concat([net, action], 1), 200, activation=tf.nn.relu, reuse=reuse)
			net = tf.contrib.layers.layer_norm(net)
		with tf.variable_scope(scope+'/dense4'):
			preds = tf.layers.dense(net, 1, activation=None, reuse=reuse)
		return state, action, preds, next_state, reward, terminal


	def _build_optimizer(self):
		self.state, self.action, self.preds, self.next_state, self.reward, self.terminal = self._build_network()
		self.net_params = tf.trainable_variables(scope='Critic')

		v_next_state = self.value_net.predict(self.next_state)
		q_target = self.reward + self.gamma*(1-self.terminal)*v_next_state
		Jq = 0.5*tf.reduce_mean((self.preds - q_target)**2.0)
		del_Jqs = [tf.gradients(xs=param, ys=Jq) for param in self.net_params]
		for i in range(len(self.net_params)):
			self.net_params[i].assign(self.net_params[i] - lambda_q*del_Jqs[i])
		self.train_op = self.net_params

	def train(self, *args):
		# args (state, action, next_state, reward, terminal)
		return self.sess.run([self.preds, self.train_op], feed_dict={
			self.state: args[0],
			self.action: args[1],
			self.next_state: args[2],
			self.reward: args[3],
			self.terminal: args[4]
			})

	def predict(self, *args):
		# args (state, action)
		return self.sess.run([self.preds], feed_dict={
			self.state: args[0],
			self.action:args[1]
			})



class ActorNetwork(object):
	#value_net is a ValueNetwork object
	def __init__(self, sess, state_dim, action_dim, value_net, critic_net):
		pass

	def _build_network(self, scope='Actor', reuse=False):
		pass

	def _build_optimizer(self):
