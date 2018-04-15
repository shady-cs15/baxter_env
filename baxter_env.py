from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np

class baxter_env(Env):
	def compute_reward(cur_goal, goal, sparse=False):
		assert cur_goal.shape == goal.shape
		return np.linalg.norm(cur_goal - goal, axis=-1)

	def set_goal_state(self, goal_state):
		self.goal_state = goal_state

	def set_limb(self, limb, id='right'):
		self.limb = limb
		self.joints = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']
		if id=='left':
			self.joints = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
		
	# returns next_state,reward, done
	def step(self, action):
		eps = 0.01
		joints = self.joints
		
		angles = self.limb.joint_angles()
		for i in range(len(joints)):
			angles[joints[i]] = action[i]
		self.limb.move_to_joint_positions(angles)
		self._state = list(self.limb.endpoint_pose()['position'])

		reward = - np.sum(np.square(np.array(self._state) - np.array(self.goal_state)))**0.5

		done = True

		for x in range(3):
			if abs(self._state[x] - self.goal_state[x])>eps:
				done = False

		next_observation = np.copy(self._state)
		print('step taken, reward:', reward)
		return Step(observation=next_observation, reward=reward, done=done)

	# resets and returns state
	def reset(self):
		action = np.random.uniform(-3.059, 3.059, size=(7,))
		joints = self.joints
		angles = self.limb.joint_angles()
		for i in range(len(joints)):
			angles[joints[i]] = action[i]
		self.limb.move_to_joint_positions(angles)
		self._state = list(self.limb.endpoint_pose()['position'])

		observation = np.copy(self._state)
		return observation

	@property
	def action_space(self):
		return Box(low=-0.1, high=0.1, shape=(7,))

	@property
	def observation_space(self):
		return Box(low=-100, high=100, shape=(3,))

	def render(self):
		print('current state:', self._state)
