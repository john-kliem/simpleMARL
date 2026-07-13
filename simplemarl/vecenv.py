import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
from gymnasium.spaces import Discrete, Box
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory


def _make_shm(shm_blocks, shape, dtype):
    """Allocate one shared memory block, register it, and return a local ndarray view."""
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_blocks.append(shm)
    view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    conf = {'name': shm.name, 'shape': shape, 'dtype': dtype}
    return view, conf


def worker_pettingzoo_zerocopy(conn, env_fn, num_envs, start_idx, shm_config):
    envs = [env_fn() for _ in range(num_envs)]
    agents = envs[0].agents

    shm_objs = []
    shms = {aid: {} for aid in agents}
    for aid in agents:
        for key, conf in shm_config[aid].items():
            shm = shared_memory.SharedMemory(name=conf['name'])
            shm_objs.append(shm)
            full = np.ndarray(conf['shape'], dtype=conf['dtype'], buffer=shm.buf)
            shms[aid][key] = full[start_idx: start_idx + num_envs]

    # Single shared critic-state block, not per-agent
    state_conf = shm_config['state']
    state_shm = shared_memory.SharedMemory(name=state_conf['name'])
    shm_objs.append(state_shm)
    state_full = np.ndarray(state_conf['shape'], dtype=state_conf['dtype'], buffer=state_shm.buf)
    state_view = state_full[start_idx: start_idx + num_envs]

    while True:
        cmd, _ = conn.recv()
        if cmd == "close":
            break

        for i, env in enumerate(envs):
            if cmd == "step":
                actions = {aid: int(shms[aid]['actions'][i]) if isinstance(env.action_spaces[aid], Discrete)
                           else shms[aid]['actions'][i] for aid in agents}
                obs, rews, terms, truncs, state = env.step(actions)
                if any(terms.values()):
                    obs, state = env.reset()
                for aid in agents:
                    shms[aid]['rews'][i] = rews[aid]
                    shms[aid]['terms'][i] = terms[aid]
                    shms[aid]['truncs'][i] = truncs[aid]
            else:  # "reset"
                obs, state = env.reset()
                for aid in agents:
                    shms[aid]['rews'][i] = 0.0
                    shms[aid]['terms'][i] = 0.0
                    shms[aid]['truncs'][i] = 0.0

            for aid in agents:
                shms[aid]['obs'][i] = obs[aid]
            state_view[i] = state

        conn.send("Done")


class SubProcVecEnv:
    def __init__(self, env_fn, num_workers, num_envs_per_worker):
        temp_env = env_fn()
        self.agents = temp_env.agents
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker
        total_envs = num_workers * num_envs_per_worker

        self.shm_blocks = []
        self.state_views = {aid: {} for aid in self.agents}
        self.agent_rewards = {aid: np.zeros(total_envs, dtype=np.float32) for aid in self.agents}
        self.episode_lengths = np.zeros(total_envs, dtype=np.int32)

        shm_configs = {aid: {} for aid in self.agents}
        for aid in self.agents:
            specs = {
                'obs':     ((total_envs, *temp_env.observation_spaces[aid].shape), np.float32),
                'actions': ((total_envs, *temp_env.action_spaces[aid].shape),      np.float32),
                'rews':    ((total_envs,), np.float32),
                'terms':   ((total_envs,), np.bool_),
                'truncs':  ((total_envs,), np.bool_),
            }
            for key, (shape, dtype) in specs.items():
                view, conf = _make_shm(self.shm_blocks, shape, dtype)
                self.state_views[aid][key] = view
                shm_configs[aid][key] = conf

        # One shared critic-state block for all agents/workers: (total_envs, num_agents, obs_dim)
        state_shape = (total_envs, *temp_env.get_state().shape)
        self.critic_state_view, state_conf = _make_shm(self.shm_blocks, state_shape, np.float32)
        shm_configs['state'] = state_conf

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
            np.copyto(self.state_views[aid]['actions'], actions_dict[aid])
        for conn in self.conns:
            conn.send(("step", None))

    def step_wait(self):
        for conn in self.conns:
            conn.recv()
        info = {'episode_lengths': [], 'rewards': {aid: [] for aid in self.agents}}
        self.episode_lengths += 1
        for i in range(self.episode_lengths.shape[0]):
            for aid in self.agents:
                self.agent_rewards[aid][i] += self.state_views[aid]['rews'][i]
                if self.state_views[aid]['terms'][i] or self.state_views[aid]['truncs'][i]:
                    if aid == self.agents[0]:
                        info['episode_lengths'].append(self.episode_lengths[i])
                    info['rewards'][aid].append(self.agent_rewards[aid][i])
                    self.episode_lengths[i] = 0
                    self.agent_rewards[aid][i] = 0.0
        return self.state_views, self.critic_state_view, info

    def reset(self):
        for conn in self.conns:
            conn.send(("reset", None))
        for conn in self.conns:
            conn.recv()
        self.episode_lengths.fill(0)
        for aid in self.agents:
            self.agent_rewards[aid].fill(0)
        return self.state_views, self.critic_state_view

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
