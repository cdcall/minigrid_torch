from multiprocessing import Process, Pipe
from utils.storage import *


def worker(conn, env):

    while True:
        cmd, data, logger = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data, logger)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError


class ParallelEnv:
    """
        A concurrent execution of environments in multiple processes.
        RL version supports logging per environment
    """

    def __init__(self, envs, log_dir=None):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        self.loggers = []
        proc_count = 0

        for env in self.envs[1:]:
            if log_dir:
                log_name = str(str(proc_count) + "_log.txt")
                step_logger = utils.get_txt_logger(log_dir, log_name)
                self.loggers.append(step_logger)
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            proc_count += 1

    def reset(self):
        for local in self.locals:
            local.send(("reset", None, None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions, logger=None):
        if self.loggers:
            for local, logger, action in zip(self.locals, self.loggers, actions[1:]):
                local.send(("step", action, logger))
            obs, reward, done, info = self.envs[0].step(actions[0], self.loggers[0])
        else:
            for local, action in zip(self.locals, actions[1:]):
                local.send(("step", action))
            obs, reward, done, info = self.envs[0].step(actions[0])

        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results
