from rllab.misc import ext

### Adding an extra buffer for demos

'''Create SimpleReplayPool instance
'''
# demo_buffer = SimpleReplayPool(max_pool_size=.., 
#	observation_dim = ..,
#	action_dim = ..)

'''adding samples
'''
# demo_buffer.add_sample(observation, action, reward, terminal)

'''getting random batch from demo_buffer
'''
# demo_batch = demo_buffer.random_batch(batch_size)



### Changes to make inside ddpg.py DDPG.init_opt()

'''create bcloss line:286
'''
demo_obs = self.env.observation_space.new_tensor_variable('demo_obs', extra_dims=1)
demo_actions = self.env.observation_space.new_tensor_variable('demo_actions', extra_dims=1)
a_ = self.policy.get_action_sym(demo_obs)
mask = self.qf.get_qval_sym(demo_obs, demo_actions) > self.qf.get_qval_sym(demo_obs, a_, deterministic=True)
bcloss = TT.mean(TT.sum((a - a_)**2.0, axis=1)*mask)


'''subtract bcloss to exisiting policy loss line:303 (check once)
'''
policy_loss = ...


'''Change f_train_policy line:318
'''
f_train_policy = ext.compile_function(
	inputs=[obs, demo_obs, demo_actions],
	outputs=policy_loss,
	updates=policy_updates
	)
	


### Changes to make inside ddpg.py DDPG.do_training()
'''Get demonstrations from demo_batch line:338
'''
demo_obs, demo_actions, _, _, _ = ext.extract(
            demo_batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )


'''update inputs in f_train_policy line:353
'''
policy_surr = f_train_policy(obs, demo_obs, demo_actions)