from rllab.algos.base import RLAlgorithm
from rllab.algos.ddpg import SimpleReplayPool
from rllab.algos.ddpg import DDPG
from rllab.misc.overrides import overrides
from rllab.misc import special
from rllab.misc import ext
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
from functools import partial
import rllab.misc.logger as logger
import theano.tensor as TT
import pickle as pickle
import numpy as np
import pyprind
import lasagne


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **ext.compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **ext.compact(kwargs))
    else:
        raise NotImplementedError

class DDPGHER(DDPG):
    
    def strategy(episode, type='final'):
    	if type=='final':
    		return episode[-1]
    	else:
    		raise NotImplementedError

    def concat(observation, goal):
    	# concatenate observation and goal
    	raise NotImplementedError

    @overrides
    def train(self):
        # This seems like a rather sequential method
        pool = SimpleReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=self.env.observation_space.flat_dim,
            action_dim=self.env.action_space.flat_dim,
        )
        self.start_worker()

        self.init_opt()
        itr = 0
        path_length = 0
        path_return = 0
        terminal = False
        observation = self.env.reset()

        sample_policy = pickle.loads(pickle.dumps(self.policy))

        ### initialise transitions and episodes
        _transitions = []
        _episode = []
        goal = None

        for epoch in range(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):
                if terminal:
                    observation = self.env.reset()
                    self.es.reset()
                    sample_policy.reset()
                    self.es_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0

                    ### add into pool
                    ### env.set_goal_state and env.get_goal_state
                    ### needs to be defined
                    for t in range(len(_transitions)):
                    	observation, action, reward, terminal, goal = _transitions[t]
                    	observation_goal = concat(observation, goal)
                    	pool.add_sample(observation_goal, action, reward, terminal)
                    	G = strategy(_episode)
                    	for g in G:
                    		self.env.set_goal_state(g)
                    		_, reward, _, _, _ = self.env.step(action)
                    		observation_goal = concat(observation, g)
                    		pool.add_sample(observation_goal, action, reward, terminal)
                    	
                    	

                    ### reset transitions and episodes
                    _transitions = []
                    _episode = []


                action = self.es.get_action(itr, observation, policy=sample_policy)

                next_observation, reward, terminal, _ = self.env.step(action)
                path_length += 1
                path_return += reward

                goal = self.env.get_goal_state()
                if goal is None:
                	print 'set the goal to the current goal in the env'
                	raise NotImplementedError

                ### store transitions and observations
                if not terminal and path_length >= self.max_path_length:
                    terminal = True
                    if self.include_horizon_terminal_transitions:
                        #pool.add_sample(observation, action, reward * self.scale_reward, terminal)
                        _transitions.append((observation, action, reward * self.scale_reward, terminal, goal))
                        _episode.append(observation)
                else:
                    #pool.add_sample(observation, action, reward * self.scale_reward, terminal)
                    _transitions.append((observation, action, reward * self.scale_reward, terminal, goal))
                    _episode.append(observation)

                observation = next_observation

                if pool.size >= self.min_pool_size:
                    for update_itr in range(self.n_updates_per_sample):
                        batch = pool.random_batch(self.batch_size)
                        self.do_training(itr, batch)
                    sample_policy.set_param_values(self.policy.get_param_values())

                itr += 1

            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                self.evaluate(epoch, pool)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                self.update_plot()
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.env.terminate()
        self.policy.terminate()
    