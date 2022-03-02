import gym
import numpy as np

class CartPoleEncoding(gym.ObservationWrapper):
    
    def observation(self, obs):

        ## Scaled Encoding
        # Scale cart position (range [-4.8, 4.8]) to range [0, 2pi]
        obs[0] = ((obs[0] + 4.8) / 9.6) * 2 * np.pi
        
        # Scale pole angle (range [-0.418, 0.418]) to range [0, 2pi]
        obs[2] = ((obs[2] + 0.418) / 0.836) * 2 * np.pi
        
        ## Continuous Encoding
        obs[1] = np.arctan(obs[1])
        obs[3] = np.arctan(obs[3])
        
        return obs