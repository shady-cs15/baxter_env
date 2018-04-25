

# demo_batch is obtained from SimpleRelayPool.random_batch
def bc_loss(demo_batch, Q, pi):
	observations = demo_batch['observations']
	actions = demo_batch['actions']
	assert actions.shape[0] == observations.shape[0]

	# actions : batch_size x action_dim
	# observations : batch_size x observ_dim
	
	loss = 0.
	for i in len(range(actions.shape[0])):
		ai = actions[i]
		si = observations[i]
		mask = 0.
		if Q(si, ai) > Q(si, pi(si)):	mask = 1.
		loss = loss + mask*(pi(si)-ai)**2.0
	return loss
