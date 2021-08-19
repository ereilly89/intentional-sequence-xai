import gym
import gym_minigrid

def make_env(env_key, seed=None):
    print("making env...")
    env = gym.make(env_key)
    print("...")
    env.seed(seed)
    return env
