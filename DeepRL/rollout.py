import numpy as np
import pickle

from platooning_env import PlatooningEnv

ENV_PARAMS = {
    # ======= REQUIRED PARAMS =======

    # reward function, one of "space_headway", "time_headway"
    "reward_type": "time_headway",

    # desired time or space headway, depending on the "reward_type" term
    "desired_headway": 1,

    # maximum torque
    "max_torque": 1,  # TODO: get good value

    # maximum break pressure
    "max_break": 1,  # TODO: get good value

    # time horizon
    "horizon": 1500,

    # simulation step size
    "sim_step": 0.1,

    # total number of trucks (including the leader)
    "num_trucks": 3,

    # ======= OPTIONAL PARAMS =======

    # std of noise associated with the headways (in m), defaults to 0
    "headway_range": [0, 50],  # TODO: get good value
    "headway_accuracy": 0,  # TODO: get good value

    # std of noise associated with the speed (in m/s), default to 0
    "speed_range": [0, 30],  # TODO: get good value
    "speed_accuracy": 0,  # TODO: get good value

    # std of noise associated with the acceleration (in m/s^2), default to 0
    "accel_range": [-5, 5],  # TODO: get good value
    "accel_accuracy": 0,  # TODO: get good value

    # observation delay (in sec), defaults to 0
    "observation_delay": 0,  # TODO: get good value

    # std of noise associated with the acceleration actions (in m/s^2),
    # defaults to 0
    "action_noise": 0,  # TODO: get a good value

    # acceleration delay (in sec), defaults to 0
    "action_delay": 0,  # TODO: get a good value

    # probability of a missing observation (defaults to 0)
    "prob_missing_obs": 0,  # TODO: get a good value
}


if __name__ == "__main__":
    # initialize the environment
    env = PlatooningEnv(env_params=ENV_PARAMS)

    num_rollouts = 2
    states = []
    for _ in range(num_rollouts):
        # reset the environment
        env.reset()

        # advance the experiment until a terminating condition is met
        done = False
        total_reward = 0
        while not done:
            action = np.random.uniform(
                -abs(ENV_PARAMS["max_break"]), abs(ENV_PARAMS["max_torque"]), 2)
            state, reward, done, _ = env.step(action)
            states.append(state)
            total_reward += reward
        print("total reward: {}".format(total_reward))

    # save the states from the rollout
    pickle.dump(states, open("observations.pkl", "wb"))
