
from Framework.ppo import PPOAgent
from collections import namedtuple
import gym

# A quick run test for debugging
env = gym.make('Pendulum-v1')
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape
model = PPOAgent(obs_dim, act_dim)

# If actor and critic have undergone previous training, load them here
# model.load('your path here')

# train it with this many iterations
for epoch in range(10000):
    for rollout in range(5):
        # collect transitions here
        transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
        model.store_transition(transition)
    model.train_step()

model.save('your path here')


