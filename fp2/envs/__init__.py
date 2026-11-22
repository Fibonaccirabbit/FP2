"""Multi-objective MuJoCo environments from PEDA (https://github.com/baitingzbt/PEDA, MIT License, Copyright (c) 2023 baitingzbt)."""
from gym.envs.registration import register

# The six D4MORL tasks (MO-Hopper-v3 has 3 objectives, the others 2).
TASKS = ["MO-Ant-v2", "MO-HalfCheetah-v2", "MO-Hopper-v2", "MO-Hopper-v3", "MO-Swimmer-v2", "MO-Walker2d-v2"]

for env_id, entry_point in [
    ("MO-Ant-v2", "fp2.envs.ant:AntEnv"),
    ("MO-HalfCheetah-v2", "fp2.envs.half_cheetah:HalfCheetahEnv"),
    ("MO-Hopper-v2", "fp2.envs.hopper:HopperEnv"),
    ("MO-Hopper-v3", "fp2.envs.hopper_v3:HopperEnv"),
    ("MO-Swimmer-v2", "fp2.envs.swimmer:SwimmerEnv"),
    ("MO-Walker2d-v2", "fp2.envs.walker2d:Walker2dEnv"),
]:
    register(id=env_id, entry_point=entry_point, max_episode_steps=500)
