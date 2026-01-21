import multiprocessing as mp
import numpy as np




def worker_pettingzoo(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, rews, terms, truncs, infos = env.step(data)
            if terms[env.agents[0]]:
                obs,_ = env.reset()
            conn.send({aid:{'obs':obs[aid],'rews':rews[aid],'terms':terms[aid],'truncs':truncs[aid],'infos':[infos]} for aid in obs})
        elif cmd == "reset":
            obs,_ = env.reset()
            conn.send({'obs':obs[aid] for aid in obs})
        #elif cmd == "state":
        #    return env.get_state()
        else:
            print("Not Implemented")
    return





class ParallelVecEnv():
    def __init__(self, env_fn, num_envs):
        assert env_fn != None, "The environment must be defined"
        assert num_envs >= 1, "Must have atleast one environment instance"
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.envs = [self.env_fn for i in range(self.num_envs)]
        #TODO: Add check if env is pettingzoo or gymansium
        self.observation_spaces = self.envs[0].observation_spaces
        self.action_spaces = self.envs[0].action_spaces
        self.state = {aid:{"obs":np.zeros((self.num_envs, self.observation_spaces[aid].shape), dtype=np.float32),
                           "rews":np.zeros((self.num_envs), dtype=np.float32),
                           "truncs":np.zeros((self.num_envs),dtype=np.float32),
                           "terms":np.zeros((self.num_envs), dtype=np.bool),
                           "info":{}} for aid in self.envs[0].agents}
        self.locals = []
        for env in self.envs:
            local, remote = mp.Pipe()
            self.locals.append(local)
            p = mp.Process(target=worker_pettingzoo, args=(remote,env))
            p.daemon=True
            p.start()
            remote.close()

        def reset(self):
            for local in self.locals:
                local.send(("reset", None))
            results = [local.recv() for local in self.locals]
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
                        self.state[aid][e][i] = result[aid][e]
            return self.state


class SerialVecEnv():
    def __init__(self, env_fn, num_envs):
        assert env_fn != None, "The environment must be defined"
        assert num_envs >= 1, "Must have atleast one environment instance"
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.envs = [self.env_fn() for i in range(self.num_envs)]
        #TODO: Add check if env is pettingzoo or gymansium
        self.observation_spaces = self.envs[0].observation_spaces
        self.action_spaces = self.envs[0].action_spaces
        self.state = None
        self.set_state()
        self.actions = None
        
        def set_state(self):
            if self.state == None:
                self.state = {aid:{"obs":np.zeros((self.num_envs, self.observation_spaces[aid].shape), dtype=np.float32),
                            "rews":np.zeros((self.num_envs), dtype=np.float32),
                            "truncs":np.zeros((self.num_envs),dtype=np.float32),
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
            results = {aid:{"obs":np.zeros(self.num_envs, self.observation_spaces[aid].shape),"info":{}} for aid in self.envs[0].agents}
            for i,e in enumerate(self.envs):
                obs, _ = e.reset()
                for aid in obs:
                    results[aid][i] = obs[aid] 
            return results

        def step_async(self, actions):
            self.actions = actions
        
        def step_wait(self):
            #TODO: Process into per agent pet format
            assert self.actions != None, "You must call step_async first"
            for i, e in enumerate(self.envs):
                obs, rew, term, trunc, info = self.envs.step({aid:self.actions[aid][i] for aid in self.actions[i]})
                if term.keys()[0]:
                    #TODO add in state passing
                    obs,_ = self.envs[i].reset()
                for aid in obs:
                    self.state[aid]["obs"][i] = obs[aid]
                    self.state[aid]["rews"][i] = rew[aid]
                    self.state[aid]["terms"][i] = term[aid]
                    self.state[aid]["truncs"][i] = trunc[aid]
                    self.state[aid]["info"][i] = info
            return self.state