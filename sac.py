import tensorflow as tf

class ValueNetwork(object):
	def __init__(self, sess, state_dim, lambda_v, tau, scope='psi'):
		self.state_dim = state_dim
		self.lambda_v = lambda_v
		self.tau = tau
		self.sess = sess
		self.scope = scope
		self._build_optimizer()

	def _build_network(self, state, reuse=False):
		scope = self.scope
		with tf.variable_scope(scope+'/dense1'):
			net = tf.layers.dense(state, 400, activation=tf.nn.relu, reuse=reuse)
			#net = tf.contrib.layers.layer_norm(net, reuse=reuse)
		with tf.variable_scope(scope+'/dense2'):
			net = tf.layers.dense(net, 300, activation=tf.nn.relu, reuse=reuse)
			#net = tf.contrib.layers.layer_norm(net, reuse=reuse)
		with tf.variable_scope(scope+'/dense3'):
			net = tf.layers.dense(net, 200, activation=tf.nn.relu, reuse=reuse)
			#net = tf.contrib.layers.layer_norm(net, reuse=reuse)
		with tf.variable_scope(scope+'/dense4'):
			preds = tf.layers.dense(net, 1, activation=None, reuse=reuse)
		return state, preds

	def _build_optimizer(self):
		state = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		self.target = tf.placeholder(tf.float32, shape=[None, 1])
		self.state, self.preds = self._build_network(state)
		self.net_params = tf.trainable_variables(scope=self.scope)

		self.Jv = 0.5*(tf.reduce_mean(self.preds-self.target)**2.0)
		del_Jvs = [tf.gradients(xs=param, ys=self.Jv)[0] for param in self.net_params]
		self.train_ops =[]
		for i in range(len(self.net_params)):
			self.train_ops.append(self.net_params[i].assign(self.net_params[i] - self.lambda_v*del_Jvs[i]))

	def predict(self, *args):
		# args (state)
		return self.sess.run([self.preds], feed_dict={
			self.state:args[0]
			})
		
	def train(self, *args):
		# args (state, target)
		# target is q(s, a) - logpi(s)
		return self.sess.run([self.preds, self.Jv, self.train_ops], feed_dict={
			self.state: args[0],
			self.target: args[1]
			})

	def update(self, value_net):
		psi = self.sess.run(value_net.net_params)
		psi_ = self.sess.run(self.net_params)
		for i in range(len(psi_)):
			psi_[i] = self.tau*psi[i] + (1-self.tau)*psi_[i]

		for i in range(len(psi_)):
			self.sess.run(self.net_params[i].assign(psi_[i]))
				
class CriticNetwork(object):
	# value_net is a ValueNetwork object
	def __init__(self, sess, state_dim, action_dim, value_net, gamma, lambda_q, scope='theta'):
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.value_net = value_net
		self.gamma = gamma
		self.lambda_q = lambda_q
		self.scope = scope
		self._build_optimizer()

	def _build_network(self, state, action, reuse=False):
		scope = self.scope
		next_state = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		reward = tf.placeholder(tf.float32, shape=[None, 1])
		terminal = tf.placeholder(tf.float32, shape=[None, 1])
		with tf.variable_scope(scope+'/dense1'):
			net = tf.layers.dense(state, 400, activation=tf.nn.relu, reuse=reuse)
			#net = tf.contrib.layers.layer_norm(net, reuse=reuse)
		with tf.variable_scope(scope+'/dense2'):
			net = tf.layers.dense(tf.concat([net, action], 1), 300, activation=tf.nn.relu, reuse=reuse)
			#net = tf.contrib.layers.layer_norm(net, reuse=reuse)
		with tf.variable_scope(scope+'/dense3'):
			net = tf.layers.dense(tf.concat([net, action], 1), 200, activation=tf.nn.relu, reuse=reuse)
			#net = tf.contrib.layers.layer_norm(net, reuse=reuse)
		with tf.variable_scope(scope+'/dense4'):
			preds = tf.layers.dense(net, 1, activation=None, reuse=reuse)
		return state, action, preds, next_state, reward, terminal


	def _build_optimizer(self):
		state = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		action = tf.placeholder(tf.float32, shape=[None, self.action_dim])
		self.state, self.action, self.preds, self.next_state, self.reward, self.terminal = self._build_network(state, action)
		self.net_params = tf.trainable_variables(scope=self.scope)

		v_next_state, _ = self.value_net._build_network(self.next_state, reuse=True)
		q_target = self.reward + self.gamma*(1-self.terminal)*v_next_state
		self.Jq = 0.5*tf.reduce_mean((self.preds - q_target)**2.0)
		del_Jqs = [tf.gradients(xs=param, ys=self.Jq)[0] for param in self.net_params]
		self.train_ops = []
		for i in range(len(self.net_params)):
			self.train_ops.append(self.net_params[i].assign(self.net_params[i] - self.lambda_q*del_Jqs[i]))

	def train(self, *args):
		# args (state, action, next_state, reward, terminal)
		return self.sess.run([self.preds, self.Jq, self.train_ops], feed_dict={
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
	# TO NOTE: policy is a gmm policy
	def __init__(self, sess, state_dim, action_dim, value_net, critic_net, lambda_pi):
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.value_net = value_net
		self.critic_net = critic_net

	def _build_network(self, state, scope='Actor', reuse=False):
		# action needs to be sampled using GMM
		raise NotImplementedError

	def _build_optimizer(self):
		state = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		self.state, self.action = self._build_network(state)

		logpi_target_q = self.critic_net._build_network(self.state, self.action, reuse=True)
		logpi_target_v = self.value_net._build_network(self.state, reuse=True)
		logpi_target = logpi_target_q - logpi_target_v
		logpi = tf.log(self.action)
		del_Jpis = [tf.gradients(xs=param, ys=logpi)*(logpi - logpi_target) for param in self.net_params]
		for i in range(len(self.net_params)):
			self.net_params[i].assign(self.net_params[i] - lambda_pi*del_Jpis[i])
		self.train_op = self.net_params

	def train(self, *args):
		# args (state)
		# Note Jpi is intractable because of partition function
		return self.sess.run([self.action, self.train_op], feed_dict={
			self.state: args[0]
		})

	def predict(self, *args):
		pass
		# args (state)
		#return self.sess.run([self.action], feed_dict={
		#	self.state: args[0]
		#	})
