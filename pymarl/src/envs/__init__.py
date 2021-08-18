from .snakes import *
from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.snake_env import SnakeEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["snake"] = partial(env_fn, env=SnakeEnv)
