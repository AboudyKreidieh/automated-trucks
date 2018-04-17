from rllab.sampler.utils import rollout
import argparse
import joblib
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of rollouts we will average over')

    args = parser.parse_args()

    # extract the flow environment
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']

    # Recreate experiment params
    max_path_length = int(np.floor(env.horizon))

    # Load data into arrays
    rew = []
    for j in range(args.num_rollouts):
        # run a single rollout of the experiment
        path = rollout(env=env,
                       agent=policy,
                       max_path_length=max_path_length)

        # collect the observations and rewards from the rollout
        new_rewards = path['rewards']

        # print the cumulative reward of the most recent rollout
        print("Round {}, return: {}".format(j, sum(new_rewards)))
        rew.append(sum(new_rewards))

    # print the average cumulative reward across rollouts
    print("Average return: {}".format(np.mean(rew)))
