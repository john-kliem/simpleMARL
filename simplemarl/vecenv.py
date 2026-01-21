import multiprocessing as mp
import numpy as np



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
        
def make_env_train():
    def thunk():
        env = MaritimeRaceEnv()
if __name__ == "__main__":    
    from maritime_env import MaritimeRaceEnv
    import time
    num_envs = 5
    num_steps = 6000
    sve = SerialVecEnv(make_env_train, num_envs)
    rets = sve.reset()
    
    action = {"agent_0":[0 for i in range(num_envs)]}
    s = time.perf_counter()
    for i in range(1200):
        sve.step_async(action)
        rets = sve.step_wait()
        # for aid in rets:
        #     print("Agent: ", aid)
        #     for k in rets[aid]:
        #         print("K: ",k," Value: ", rets[aid][k])
    sve.close()
    print("100 Envs Serial: ", time.perf_counter() - s)
    #Check Parallel Environment
    pve = ParallelVecEnv(make_env_train, num_envs)
    rets = pve.reset()
   
    action = {"agent_0":[0 for i in range(num_envs)]}
    s = time.perf_counter()
    for i in range(1200):
        # print("Step: ",i)
        pve.step_async(action)
        rets = pve.step_wait()
        # print("Final Rets: ", rets)
        # for aid in rets:
            # print("Agent: ", aid)
            # for k in rets[aid]:
                # print("K: ",k," Value: ", rets[aid][k])
    print("100 envs Parallel: ", time.perf_counter()-s)
    pve.close()

