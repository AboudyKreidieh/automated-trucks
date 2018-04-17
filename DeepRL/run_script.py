"""Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
merges in an open network.

"""
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.envs.gym_env import GymEnv
from gym.envs.registration import register

from rollout import ENV_PARAMS

HORIZON = 1500
ENV_PARAMS["horizon"] = HORIZON


def run_task(_):
    env_name = "PlatooningEnv"

    register(
        id=env_name+'-v0',
        entry_point='platooning_env:{}'.format(env_name),
        max_episode_steps=HORIZON,
        kwargs={"env_params": ENV_PARAMS}
    )

    env = GymEnv(env_name, record_video=False)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(16, 16, 16),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=15000,
        max_path_length=horizon,
        n_itr=1000,
        # whole_paths=True,
        discount=0.999,
    )
    algo.train(),

exp_tag = "2-car-platooning"

for seed in [5]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Keeps the snapshot parameters for all iterations
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        # plot=True,
    )
