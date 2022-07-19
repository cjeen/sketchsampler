import os
from typing import Optional

import dotenv
import numpy as np


def get_classdict(dataset='shapenet'):
    if dataset == 'shapenet':
        class_dict = {'04379243': 'table', '02958343': 'car', '03001627': 'chair',
                      '02691156': 'plane', '04256520': 'couch', '04090263': 'firearm',
                      '03636649': 'lamp', '04530566': 'watercraft', '02828884': 'bench',
                      '03691459': 'speaker', '02933112': 'cabinet', '03211117': 'monitor',
                      '04401088': 'cellphone'}
    else:
        raise NotImplementedError
    return class_dict


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])[..., None]


def get_env(env_name: str) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        raise KeyError(f"{env_name} not defined")
    env_value: str = os.environ[env_name]
    if not env_value:
        raise ValueError(f"{env_name} has yet to be configured")
    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


load_envs()
PROJECT_ROOT = get_env("PROJECT_ROOT")
