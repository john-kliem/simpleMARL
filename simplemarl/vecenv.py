import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
from gymnasium.spaces import Discrete, Box
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory

def worker_pettingzoo_zerocopy(conn, env_fn, num_envs, start_idx, shm_config):
    # Create the local sub-environments for this worker
    envs = [env_fn() for _ in range(num_envs)]
    agents = envs[0].agents
    
    # Map to shared memory
    shm_objs = []
    shms = {aid: {} for aid in agents}
    for aid in agents:
        for key, conf in shm_config[aid].items():
            # Attach to the FULL block
           
            shm = shared_memory.SharedMemory(name=conf['name'])
            shm_objs.append(shm)
            full_block = np.ndarray(conf['shape'], dtype=conf['dtype'], buffer=shm.buf)
            # Create a VIEW of just this worker's assigned rows
            shms[aid][key] = full_block[start_idx : start_idx + num_envs]
    while True:
        cmd, _ = conn.recv()
        if cmd == "step":
            for i, env in enumerate(envs):
                # The worker reads actions directly from its shared view
                # Actions were placed there by the main process
                actions = {aid: int(shms[aid]['actions'][i]) if isinstance(env.action_spaces[aid], Discrete) else shms[aid]['actions'][i] for aid in agents}
                #Check state for terms
                obs, rews, terms, truncs, _ = env.step(actions)
                
                if any(terms.values()) or any(truncs.values()):
                    obs, _ = env.reset()
                for aid in agents:
                    shms[aid]['obs'][i] = obs[aid]
                    shms[aid]['rews'][i] = rews[aid] 
                    shms[aid]['terms'][i] = terms[aid]
                    shms[aid]['truncs'][i] = truncs[aid]
            
            conn.send("Done")
        elif cmd == "reset":
            for i,env in enumerate(envs):
                obs,_ = env.reset()
                for aid in obs:
                    shms[aid]['obs'][i] = obs[aid]
                    shms[aid]['terms'][i] = 0.0
                    shms[aid]['terms'][i] = 0.0 
                    shms[aid]['rews'][i] = 0.0
            conn.send("Done")
        elif cmd == "close":
            break

class SubProcVecEnv:
    def __init__(self, env_fn, num_workers, num_envs_per_worker):
        temp_env = env_fn()
        self.agents = temp_env.agents
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker
        total_envs = num_workers * num_envs_per_worker
        
        self.shm_blocks = []
        self.state_views = {aid: {} for aid in self.agents}
        shm_configs = {aid: {} for aid in self.agents}

        for aid in self.agents:
            specs = {
                'obs': (total_envs, *temp_env.observation_spaces[aid].shape),
                'rews': (total_envs,),
                'terms': (total_envs,),
                'truncs': (total_envs,),
                'actions': (total_envs, *temp_env.action_spaces[aid].shape)
            }

            for key, shape in specs.items():
                # Corrected type check
                dtype = np.float32 if key in ['obs', 'rews', 'actions'] else np.bool_
                nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
                
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self.shm_blocks.append(shm)
                
                self.state_views[aid][key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                shm_configs[aid][key] = {'name': shm.name, 'shape': shape, 'dtype': dtype}

        self.conns = []
        for i in range(num_workers):
            parent_conn, child_conn = mp.Pipe()
            start_idx = i * num_envs_per_worker
            p = mp.Process(target=worker_pettingzoo_zerocopy, 
                           args=(child_conn, env_fn, num_envs_per_worker, start_idx, shm_configs))
            p.daemon = True
            p.start()
            self.conns.append(parent_conn)

    def step_async(self, actions_dict):
        for aid in self.agents:
            # Direct copy of all actions into the shared memory view
            np.copyto(self.state_views[aid]['actions'], actions_dict[aid])
        
        for conn in self.conns:
            conn.send(("step", None))
        
    def step_wait(self):
        for conn in self.conns:
            conn.recv()
        return self.state_views
    def reset(self):
        for conn in self.conns:
            conn.send(("reset", None))
        for conn in self.conns:
            conn.recv()
        return self.state_views
    def close(self):
        for conn in self.conns:
            conn.send(("close", None))
        for shm in self.shm_blocks:
            shm.close()
            shm.unlink()
    

#TODO Pass in memory space and directly assign values to shared space
def worker_pettingzoo(conn, env):
    env = env()
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, rews, terms, truncs, infos = env.step(data)
            if terms[env.agents[0]]:
                obs,_ = env.reset()
            conn.send({aid:{'obs':obs[aid],'rews':rews[aid],'terms':terms[aid],'truncs':truncs[aid],'infos':infos[aid] if aid in infos else {}} for aid in obs})
        elif cmd == "reset":
            
            obs,_ = env.reset()
            conn.send({aid:{'obs':obs[aid], 'info':{}} for aid in obs})
        #elif cmd == "state":
        #    return env.get_state()
        elif cmd == "close":
            env.close()
            return
        else:
            print("Not Implemented")
    return
#TODO Write a version to be subproc (SubProcVecEnv) with shared memory map No passing large datastructures over pipes
class ParallelVecEnv():
    #Parallel Env where one environment per process and isn't using a shared memory map
    def __init__(self, env_fn, num_envs):
        assert env_fn != None, "The environment must be defined"
        assert num_envs >= 1, "Must have atleast one environment instance"
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.envs = [self.env_fn for i in range(self.num_envs)]
        temp_env = self.env_fn()
        self.agents = temp_env.agents
        #TODO: Add check if env is pettingzoo or gymansium
        self.observation_spaces = temp_env.observation_spaces
        self.action_spaces = temp_env.action_spaces
        self.state = {aid:{"obs":np.zeros((self.num_envs, *self.observation_spaces[aid].shape), dtype=np.float32),
                           "rews":np.zeros((self.num_envs), dtype=np.float32),
                           "truncs":np.zeros((self.num_envs),dtype=np.bool),
                           "terms":np.zeros((self.num_envs), dtype=np.bool),
                           "info":{}} for aid in temp_env.agents}
        self.locals = []
        for env in self.envs:
            local, remote = mp.Pipe()
            self.locals.append(local)
            p = mp.Process(target=worker_pettingzoo, args=(remote,env))
            p.daemon=True
            p.start()
            remote.close()

    def reset(self):
        results = {aid:{"obs":np.zeros((self.num_envs, *self.observation_spaces[aid].shape),dtype=np.float32),"info":{}} for aid in self.agents}
    
        for local in self.locals:
            local.send(("reset", None))
        #Collect Responses
        for i,local in enumerate(self.locals):
            recv = local.recv()
            for aid in recv:
                results[aid]["obs"][i] = recv[aid]["obs"]
        return results

    def step_async(self, actions):
        for i,local in enumerate(self.locals):
            local.send(("step",{aid:actions[aid][i] for aid in actions}))

    def step_wait(self):
        #TODO: Process into per agent pet format

        for i, local in enumerate(self.locals):
            result = local.recv()
            for aid in result:
                for e in result[aid]:
                    if e == "infos":
                        self.state[aid][e] = result[aid][e]
                    else:
                        self.state[aid][e][i] = result[aid][e]
        return self.state
    def close(self):
        for i, local in enumerate(self.locals):
            local.send(("close", None))


class SerialVecEnv():
    def __init__(self, env_fn, num_envs):
        assert env_fn != None, "The environment must be defined"
        assert num_envs >= 1, "Must have atleast one environment instance"
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.envs = [self.env_fn()() for i in range(self.num_envs)]
        print("Envs: ", self.envs[0])
        #TODO: Add check if env is pettingzoo or gymansium
        self.observation_spaces = self.envs[0].observation_spaces
        self.action_spaces = self.envs[0].action_spaces
        self.state = None
        self.set_state()
        self.actions = None
        
    def set_state(self):
        if self.state == None:
            
            self.state = {aid:{"obs":np.zeros((self.num_envs, *self.observation_spaces[aid].shape), dtype=np.float32),
                        "rews":np.zeros((self.num_envs), dtype=np.float32),
                        "truncs":np.zeros((self.num_envs),dtype=np.bool),
                        "terms":np.zeros((self.num_envs), dtype=np.bool),
                        "info":{}} for aid in self.envs[0].agents}
        else:
            for aid in self.state:
                for k in self.state[aid]:
                    if k == "info":
                        self.state[aid][k] = {}
                    else:
                        self.state[aid][k].fill(0)
    def reset(self):
        results = {aid:{"obs":np.zeros((self.num_envs, *self.observation_spaces[aid].shape),dtype=np.float32),"info":{}} for aid in self.envs[0].agents}
        for i,e in enumerate(self.envs):
            obs, _ = e.reset()
            for aid in obs:
                results[aid]["obs"][i] = obs[aid] 
        return results

    def step_async(self, actions):
        self.actions = actions
    
    def step_wait(self):
        #TODO: Process into per agent pet format
        assert self.actions != None, "You must call step_async first"
        for i, e in enumerate(self.envs):
            obs, rew, term, trunc, info = self.envs[i].step({aid:self.actions[aid][i] for aid in self.actions})
            if any(term.values()) or any(trunc.values()):
                #TODO add in state passing
                obs,_ = self.envs[i].reset()
            for aid in obs:
                self.state[aid]["obs"][i] = obs[aid]
                self.state[aid]["rews"][i] = rew[aid]
                self.state[aid]["terms"][i] = term[aid]
                self.state[aid]["truncs"][i] = trunc[aid]
                self.state[aid]["info"] = {}#[i] = info
        return self.state
    def close(self):
        for i,e in enumerate(self.envs):
            self.envs[i].close()
        self.envs = []
        self.state = None 
