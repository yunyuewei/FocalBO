import pickle as pkl
import design_bench as db
import torch
import numpy as np
from design_bench.oracles.feature_extractors.morgan_fingerprint_features import MorganFingerprintFeatures
from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
import matplotlib.pyplot as plt
    
name_to_env = {
    'AntMorphology': 'AntMorphology-Exact-v0',
    'DKittyMorphology': 'DKittyMorphology-Exact-v0'
}

name_to_full_dataset = {
    'AntMorphology': AntMorphologyDataset,
    'DKittyMorphology': DKittyMorphologyDataset
}

# task = db.make('AntMorphology-Exact-v0')
class DesignBenchFunction():
    def __init__(self, env_name='AntMorphology',  minimize=True):
        
        assert env_name in name_to_env
        self.task = db.make(name_to_env[env_name])
        self.__name__ = env_name
        self.full_task = name_to_full_dataset[env_name]()
        self.true_dim = self.task.input_size

        self.lb = self.full_task.x.min(0)
        self.ub = self.full_task.x.max(0)
        gap = self.ub-self.lb
        self.valid_idx = np.argwhere(gap>0).ravel()
        self.lb = self.lb[self.valid_idx]
        self.ub = self.ub[self.valid_idx]
        self.dim = len(self.valid_idx)
        # invalid_idx = np.argwhere(gap<=0).ravel()
        # print(gap[invalid_idx], self.task.x[0, invalid_idx])
        
        # print(valid_idx)
        # raise NotImplementedError
        # self.init_x = self.full_task.x
        # self.init_y = self.full_task.y
        # print(self.full_task.x.min(0)-self.lb, self.full_task.x.max(0)-self.ub) 
        
    def __call__(self, x):
        # x = x.reshape(1, -1)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if len(self.valid_idx) != self.true_dim:
            x = self.lb + x * (self.ub - self.lb)
            full_x = np.zeros((x.shape[0], self.true_dim))
            full_x[:, self.valid_idx] = x
            x = full_x
        y = self.task.predict(x)
        y = torch.from_numpy(y).to(dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).reshape(-1, 1)
        return y

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float
    fun = DesignBenchFunction('AntMorphology')
    # fun = DesignBenchFunction('DKittyMorphology')
    

    
    initial_X = (fun.task.x[:, fun.valid_idx]-fun.lb)/(fun.ub-fun.lb)
    print(initial_X[0])
    # initial_X = torch.from_numpy(initial_X).to(dtype=dtype, device=device)
    # initial_Y = torch.from_numpy(fun.task.y).to(dtype=dtype, device=device).reshape(-1, 1)
    # with open(f'../initial_data/{fun.__name__}', 'wb') as file: 
    #     # A new file will be created 
    #     all_var = (initial_X, initial_Y)
    #     pkl.dump(all_var, file) 
    print('\n\n\n')
    reconstruct_x = fun.lb + initial_X* (fun.ub-fun.lb)
    for i in range(10):
    # i = fun.full_task.y.ravel().argmax()
        print(np.sum(reconstruct_x[i]- fun.task.x[i]), 
                fun.task.predict(reconstruct_x[i].reshape(1, -1)).ravel()[0], 
                fun.task.predict(fun.task.x[i].reshape(1, -1)).ravel()[0],
                fun.task.y[i].ravel()[0]
                )
    print('\n\n\n')
    raise NotImplementedError
    # print(initial_X.min(), initial_X.max())
    
        
    
    # initial_Y = torch.from_numpy(fun.task.y).to(dtype=dtype, device=device).reshape(-1, 1)
    # print(initial_X.shape, initial_Y.shape)
    # initial_X, unique_idx = torch.unique(initial_X, sorted=False, return_inverse=True, dim=0)
    # print(initial_X.shape)
    # print(unique_idx)
    # raise NotImplementedError
 
    # gap = fun.ub - fun.lb
    # print(fun.full_task.x[:, gap.argmin()])
    # print(fun.lb.max(), fun.lb.min(), fun.ub.max(), fun.ub.min(), gap.max(), gap.min(), gap.argmin())
    
    # print(fun.task.y.max())
    print(fun.full_task.y.max(), fun.full_task.y.min(), (0.896)*(fun.full_task.y.max()-fun.full_task.y.min())+fun.full_task.y.min())
    # # x = 0.5* torch.ones(100, fun.dim)
    # # dikkity: 273
    # # ant:804 
    # # print(fun(x))
    
    # # x = fun.full_task.x[fun.full_task.y.ravel().argmax()]
    # x = initial_X[0]
    
    # print('check', fun(x.reshape(1, -1)), fun.task.y[0])
    # for i in range(10):
    #     x[2] = np.random.rand()*0.1
    #     print(i, x[2], fun(x.reshape(1, -1)))