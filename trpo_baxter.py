import rospy
import baxter_interface

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from baxter_env import baxter_env
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


rospy.init_node('baxter_env')
limb_id = 'right'
limb = baxter_interface.Limb(limb_id)

env = baxter_env()
env.set_goal_state([0.61496079407032, -0.6234904334113506, 0.8964731004359662])
env.set_limb(limb, limb_id)

env = normalize(env)

print('env set up')
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)

print('poicy prior set up')

baseline = LinearFeatureBaseline(env_spec=env.spec)

print('baseline created')
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()
