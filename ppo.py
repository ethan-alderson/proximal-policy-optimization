
from actor import ConvActor
from critic import MLPCritic
from torch.distributions import MultivariateNormal

class PPO:
    
    # CONSTRUCTS A PPO OBJECT FOR TRAINING
    def __init__(self, env):
        # Environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize networks, this is STEP 1 IN THE PSEUDOCODE
        self.actor = ConvActor(self.obs_dim, self.act_dim)
        self.critic = MLPCritic(self.obs_dim, 1)

        self._init_hyperparameters()

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

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

    # THE PRIMARY ALGORITHM, FOUND IN THE PSEUDOCODE
    def learn(self, total_timesteps):
        
        simulated_steps = 0 # Timesteps simulated so far
        # This loop is STEP 2 IN THE PSEUDOCODE
        while (simulated_steps < total_timesteps):
            # STEP 3 OF THE PSEUDOCODE
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
    
    # INITIALIZES HYPERPARAMETERS - COULD BE IMPLEMENTED WITH GRID SEARCH
    def _init_hyperparameters(self):
        # Random initial values to be changed later
        self.timesteps_per_batch = 4800 # timesteps per batch
        self.max_timesteps_per_episode = 1600 # timesteps per episode
        self.gamma = 0.95 # this is our discount factor

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
            obs = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews) 

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
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