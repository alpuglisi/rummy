import multiprocessing as mp
import numpy as np
import torch
import rummy_engine

def env_worker(remote, parent_remote, seed):
    parent_remote.close()
    env = rummy_engine.RummyEnv(seed)
    
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            reward, done = env.step(data)
            if done:
                env.reset() 
            remote.send((env.get_state(), env.get_legal_actions(), reward, done))
        elif cmd == 'reset':
            env.reset()
            remote.send((env.get_state(), env.get_legal_actions()))
        elif cmd == 'close':
            remote.close()
            break

class VectorizedRummyEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []
        
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            seed = np.random.randint(0, 2**31) + i
            p = mp.Process(target=env_worker, args=(work_remote, remote, seed))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = [remote.recv() for remote in self.remotes]
        states, masks = zip(*results)
        
        return torch.tensor(np.stack(states), dtype=torch.float32),                torch.tensor(np.stack(masks), dtype=torch.bool)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions.tolist()):
            remote.send(('step', action))
            
        results = [remote.recv() for remote in self.remotes]
        states, masks, rewards, dones = zip(*results)
        
        return torch.tensor(np.stack(states), dtype=torch.float32),                torch.tensor(np.stack(masks), dtype=torch.bool),                torch.tensor(rewards, dtype=torch.float32),                torch.tensor(dones, dtype=torch.bool)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
