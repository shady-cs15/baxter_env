'''soft actor critic'''

# Requires
# params: psi, psi_, theta, phi
# costs: Jv, Jq, Jpi
# hyperparams: lambda_V, lambda_Q, lambda_pi, tao
# constants: max_iters, max_episode_len, max_g_steps
# functions: policy.sample_action(state), env.next_state(state, action), reward(state, action)


'''Add the following after creating the costs Jv, Jq, Jpi in the computation graph'''
'''not inside session'''
del_Jv = tf.gradients(xs=psi, ys=Jv)
del_Jq = tf.gradients(xs=theta , ys=Jq)
del_Jpi = tf.gradients(xs=phi, ys=Jpi)

psi.assign(psi - lambda_V*del_Jv)
theta.assign(theta - lambda_Q*del_Jq)
phi.assign(phi - lambda_pi*del_Jpi)
psi_.assign(tau*psi + (1-tau)*psi_)

update_op = [psi, theta, phi, psi_]

'''create buffer'''
#_buffer = SimpleReplayPool(max_pool_size=.., observation_dim=.., action_dim=..)


'''following is inside tf.Session()'''
for _iter in range(max_iters):
	# get initial state in s
		s = env.reset()
	for t in range(max_episode_len):
		a = policy.sample_action(s)
		s_ = env.next_state(s, a)
		_buffer.add_sample(s, a, reward(s, a), s_)

	...

	'''update_op will perform the gradient descent updates'''
	for g_step in range(max_g_steps):
		_ = sess.run([update_op], feed_dict={...})