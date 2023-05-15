
# from actor import ConvActor
# from critic import MLPCritic

from Prototype_Implementation.network import FeedForwardNN
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import torch.nn as nn
import numpy as np

class PPO:
    
    # CONSTRUCTS A PPO OBJECT FOR TRAINING
    def __init__(self, env):
        # Environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize networks, this is STEP 1 IN THE PSEUDOCODE
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self._init_hyperparameters()

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        # Setting an optimizer as a field allows easy switching to different
        # optimization methods
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam (self.critic.parameters(), lr=self.lr)

    # THE PRIMARY ALGORITHM, FOUND IN THE PSEUDOCODE
    def learn(self, total_timesteps):
        
        simulated_steps = 0 # Timesteps simulated so far
        # This loop is STEP 2 IN THE PSEUDOCODE
        while (simulated_steps < total_timesteps):
            # STEP 3 OF THE PSEUDOCODE
            # RTG CALCULATION IS STEP 4, DONE IN calculate_rtgs helper
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Increment our simulated steps by the steps that we just took in our rollout call
            simulated_steps += np.sum(batch_lens)

            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP 5
            # Calculate advantages
            A_k = batch_rtgs - V.detach()

            # Normalize advantages, adding a small quantity to std dev to avoid division by 0
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # the denominator of our ratio is stored in batch_log_probs
                # Note that our evaluate method recollects log probabilities from
                # the most updated version of our actor, while batch_log_probs contains
                # contains the log probabilities of the same actions before the current
                # optimization loop
                # The below line calculates the numerator of the ratio in surrogate loss
                # pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                # Calculate the ratios for the trajectories here
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Next, we calculate surrogate loss, first collecting values
                # for both the clipped and unclipped surrogate objectives
                # note that torch.clamp is the same as the clip() function in the paper 
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                
                # We apply a negative sign here because we are trying to perform gradient ascent,
                # but we are utilizing an optimizer called Adam which performs stochastic descent
                # thus we perform descent on the negative of our function, which is mathematically
                # equivalent to ascent
                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Calculate gradients and perform backpropagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Calculate V_phi and pi_theta(a_t | s_t)    
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                critic_loss = nn.MSELoss()(V, batch_rtgs)
                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()


    # INITIALIZES HYPERPARAMETERS - COULD BE IMPLEMENTED WITH GRID SEARCH
    def _init_hyperparameters(self):
        # Random initial values to be changed later
        self.timesteps_per_batch = 4800 # timesteps per batch
        self.max_timesteps_per_episode = 1600 # timesteps per episode
        self.gamma = 0.95 # this is our discount factor
        self.n_updates_per_iteration = 5 # the number of updates per learning iteration
        self.clip = 0.2 # This is our epsilon value, recommended to be 0.2 by the paper
        self.lr = 0.005 # Our learning rate
 
    # COLLECTS TRAJECTORIES FOR TRAINING
    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0 
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            obs = self.env.reset()[0]
            done = False
            ep_count = 0

            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                ep_count += 1
                # Collect observation
                batch_obs.append(obs)

                # obs_tensor = torch.tensor(obs, dtype=torch.float)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, info = self.env.step(action)

                # print(result)

                # obs, rew, done, _ = ((1,1), 0, 0, 0)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # Collect episodic length and rewards
            batch_lens.append(ep_count + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews) 

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)

        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    # GETS AN ACTION FROM THE ACTOR NETWORK
    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()

    # COMPUTES REWARD-TO-GO, THIS IS PSEUDOCODE STEP 4
    def compute_rtgs(self, batch_rews):
    # The rewards-to-go (rtg) per episode per batch to return.
    # The shape will be (num timesteps per episode)
        batch_rtgs = []
    # Iterate through each episode backwards to maintain same order
    # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
    
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    

    # This queries our critic network to calculate value estimates for advantage estimation
    def evaluate(self, batch_obs, batch_acts):
        # Reduces our tensor to a 1 dimensional array
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs
    

