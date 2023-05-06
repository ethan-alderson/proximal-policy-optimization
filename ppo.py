
from actor import ConvActor
from critic import MLPCritic

class PPO:

    def __init__(self, env):
        # Environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize networks, this is STEP 1 IN THE PSEUDOCODE
        self.actor = ConvActor(self.obs_dim, self.act_dim)
        self.critic = MLPCritic(self.obs_dim, 1)

        self._init_hyperparameters()

    def learn(self, total_timesteps):
        
        simulated_steps = 0 # Timesteps simulated so far
        # This loop is STEP 2 IN THE PSEUDOCODE
        while (simulated_steps < total_timesteps):
            pass     
        
    def _init_hyperparameters(self):
        # Random initial values to be changed later
        self.timesteps_per_batch = 4800 # timesteps per batch
        self.max_timesteps_per_episode = 1600 # timesteps per episode

    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch