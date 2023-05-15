
from Prototype_Implementation.ppo import PPO
import gym

# A quick run test for debugging
env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)




