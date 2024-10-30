import numpy as np
import gym
import torch

def heuristic_Controller(s, w):
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a

def demo_heuristic_lander(env, w, seed):

    total_reward = 0
    steps = 0
    s = env.reset(seed=seed)[0]
    while True:
        a = heuristic_Controller(s, w)
        s, r, done, truncate, info = env.step(a)
        total_reward += r

        # if steps % 20 == 0 or done:
        #     print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
        #     print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if done or truncate:
            break
    env.close()
    return total_reward



class Lunar:
    def __init__(self, minimize=False):
        
        self.dim = 12
        self.lb = 0 * np.ones(self.dim)
        self.ub = 2 * np.ones(self.dim)
        self.minimize = minimize
        self.__name__ = 'Lunar'
        
    def __call__(self, w, env_num=10):
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()
        vals = np.zeros(env_num)
        def eval(w, seed):
            env = gym.make('LunarLander-v2')
            # env.seed(seed)
            val = demo_heuristic_lander(env, w, seed) # minimiztion
            if self.minimize:
                return -val
            else:
                return val
        vals = [eval(w, seed) for seed in range(env_num)]
        # for seed in range(env_num):
            
        
        return np.mean(vals)

def main():
    np.random.seed(0)
    f = Lunar()
    x = np.random.uniform(f.lb, f.ub)
    print('Input = {}'.format(x))
    print('Output = {}'.format(f(x)))


if __name__ == '__main__':
    main()