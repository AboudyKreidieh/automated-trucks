import math
import numpy as np
from scipy.stats import truncnorm
from collections import deque
from gym.spaces.box import Box
import gym
import os
import sys
import socket
import subprocess
from rllab.core.serializable import Serializable

VEHICLE_LENGTH = 5  # length of trucks TODO: get good value
HOST = 'localhost'  # Symbolic name meaning all available interfaces
PORT = 50005  # Arbitrary non-privileged port

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

    # ======= OPTIONAL PARAMS =======

    # std of noise associated with the headways (in m), defaults to 0
    "headway_range": [0, 50],  # TODO: get good value
    "headway_accuracy": 0.00,  # TODO: get good value

    # std of noise associated with the speed (in m/s), default to 0
    "speed_range": [0, 30],  # TODO: get good value
    "speed_accuracy": 0.00,  # TODO: get good value

    # std of noise associated with the acceleration (in m/s^2), default to 0
    "accel_range": [-5, 5],  # TODO: get good value
    "accel_accuracy": 0.00,  # TODO: get good value

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


def leader_trajectory(env):
    """Leading vehicle velocity trajectory."""
    v_avg = 10

    if env.time_counter < 100:
        vel = v_avg/10 * env.sim_step * env.time_counter
    else:
        vel = env.c * math.sin(env.tau*(env.time_counter-100)) + v_avg

    return vel


class PlatooningEnv(gym.Env, Serializable):
    """Fully observable, single agent platooning environment

    A leading vehicle is provided a trajectory, and the platooning vehicles are
    told to match the speed of the leading vehicle as well as maintain a
    desirable headway with its immediate leader.

    States
    ------
    The states the speed and acceleration(?) of the lead vehicle, as well as
    the speeds, headways, and accelerations(?) for the platooning vehicles,

    Actions
    -------
    Actions are (blank)

    Rewards
    -------
    Rewards are the two-norm from the desired space or time headway, depending
    on the "reward_type" term in EnvParams.

    Termination
    -----------

    """
    def __init__(self, env_params):
        Serializable.quick_init(self, locals())

        self.env_params = env_params
        self.num_trucks = env_params["num_trucks"]

        # simulation length parameters
        self.sim_step = env_params["sim_step"]
        self.horizon = env_params["horizon"]
        self.time_counter = 0

        # ids of the human and rl vehicles
        self.rl_ids = ["follower_%s" % i for i in range(self.num_trucks-1)]
        self.ids = ["leader"] + self.rl_ids

        # dict used to store the positions, speeds, and accelerations of all
        # vehicles
        self.vehicles = dict().fromkeys(self.ids)
        for veh_id in self.ids:
            self.vehicles[veh_id] = \
                dict().fromkeys(["pos", "headway", "speed", "accel"])

        # required additional parameters (see ENV_PARAMS)
        self.trajectory = leader_trajectory
        self.reward_type = env_params["reward_type"]
        self.desired_headway = env_params["desired_headway"]

        # acceleration delay
        self.action_delay = env_params.get("accel_delay", 0)

        # acceleration noise
        self.action_noise = env_params.get("accel_noise", 0)

        # queue of accelerations for each vehicle
        self.action_queue = dict()
        for veh_id in self.rl_ids:
            self.action_queue[veh_id] = deque()

        # velocity at the previous time step; used to compute accelerations
        self.prev_vel = dict()

        self.headway_noise = truncnorm(-0.05, 0.05)
        self.speed_noise = truncnorm(-0.05, 0.05)
        self.accel_noise = truncnorm(-0.05, 0.05)

        # tcp/ip connection with matlab
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        self.s.listen(1)

        # subprocess that calls the matlab/simulink method
        self.proc = None

        # rollout iteration number; the matlab/simulink method is called every
        # 10th iteration
        self.rollout_itr = 0

    @property
    def action_space(self):
        return Box(low=-abs(self.env_params["max_break"]),
                   high=self.env_params["max_torque"],
                   shape=(self.num_trucks - 1, ))

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(3 * self.num_trucks - 1,))

    def _step(self, action):
        self.time_counter += 1

        # Part 1 -- update the trajectory of the lead vehicle
        leader_id = "leader"
        # collect trajectory velocity
        human_vel = self.trajectory(self)
        # compute the trajectory acceleration
        human_accel = (human_vel - self.vehicles[leader_id]["speed"]) \
            / self.sim_step

        # speed and position and previous time step
        pos = self.vehicles[leader_id]["pos"]
        speed = self.vehicles[leader_id]["speed"]

        # apply this acceleration to the lead vehicle
        self.vehicles[leader_id]["pos"] = pos + speed * self.sim_step \
            + 0.5 * human_accel * self.sim_step ** 2
        self.vehicles[leader_id]["speed"] = speed + human_accel * self.sim_step
        self.vehicles[leader_id]["accel"] = human_accel

        # Part 2 -- apply the rl actions
        self._apply_rl_actions(action)

        # Part 3 -- get the next state
        state = list(self._get_state().T)

        # Part 4 -- compute the reward
        reward = self._compute_reward(action)

        # Part 5 -- check if the horizon has been reached
        done = self.time_counter >= self.horizon

        if done:
            self.conn.sendall(b"-----:-----:1\n")
            self.conn.close()

        return state, reward, done, {}

    def _apply_rl_actions(self, actions):
        command = ""

        # add accelerations to the command sent to matlab
        for action in actions:
            if action > 0:
                command += " "
            else:
                command += "-"
            command += "%.2f:" % abs(action)

        # add the command to terminate or not
        command += "0\n"

        # convert the command to a byte array
        command = command.encode()

        # send commands to matlab
        self.conn.sendall(command)

        # disconnect and try to reconnect to the instance
        self.conn.close()
        self.conn, address = self.s.accept()

        # get resulting data
        data = None
        while not data:  # TODO(ak): check if I need to do this
            data = self.conn.recv(1024)
        data = data.decode('ascii').split(":")

        # update the positions, speeds, and accelerations
        for i, veh_id in enumerate(self.rl_ids):
            self.vehicles[self.rl_ids[i]]["pos"] = float(data[3*i])
            self.vehicles[self.rl_ids[i]]["speed"] = float(data[3*i+1])
            self.vehicles[self.rl_ids[i]]["accel"] = float(data[3*i+2])

        # update the headways
        for i, veh_id in enumerate(self.rl_ids):
            self.vehicles[veh_id]["headway"] = \
                self.vehicles[self.ids[i]]["pos"] \
                - self.vehicles[self.ids[i+1]]["pos"] - VEHICLE_LENGTH

    def _compute_reward(self, actions):
        headway = []
        if self.reward_type == "time_headway":
            # compute the time headway for each truck
            headway = [self.vehicles[veh_id]["headway"]
                       / max(self.vehicles[veh_id]["speed"], 0.01)
                       for veh_id in self.rl_ids]
        elif self.reward_type == "space_headway":
            # compute the space headway for each truck
            headway = [self.vehicles[veh_id]["hdeadway"]
                       for veh_id in self.rl_ids]

        # reward proximity to desired headway
        return - np.linalg.norm(self.desired_headway - np.array(headway))

    def _get_state(self):
        # normalizers
        v_r = self.env_params["speed_range"]
        h_r = self.env_params["headway_range"]
        a_r = self.env_params["accel_range"]

        # headways for the rl vehicles (normalized with noise)
        dx = [(self.vehicles[veh_id]["headway"] - h_r[0]) / (h_r[1] - h_r[0])
              for veh_id in self.rl_ids]
        # dx = np.array(dx) + self.headway_noise.rvs(len(dx))

        # speeds of ALL vehicles (normalized with noise)
        v = [(self.vehicles[veh_id]["speed"] - v_r[0]) / (v_r[1] - v_r[0])
             for veh_id in self.ids]
        # v = np.array(v) + self.speed_noise.rvs(len(v))

        # accelerations of ALL vehicles (normalized with noise)
        accel = [(self.vehicles[veh_id]["accel"] - a_r[0]) / (a_r[1] - a_r[0])
                 for veh_id in self.ids]
        # accel = np.array(accel) + self.accel_noise.rvs(len(accel))

        return np.concatenate([dx, v, accel])

    def _reset(self):
        # reset the time counter
        self.time_counter = 0

        # choose new values for the trajectory
        self.c = np.random.uniform(low=1, high=5)
        self.tau = 1 / np.random.uniform(low=50, high=200)

        call = ["/usr/local/MATLAB/R2017a/bin/matlab",
                "-nodisplay", "-nodesktop", "-r",
                "run('/home/aboudy/Desktop/ITSC 2018/main.m')"]

        if self.rollout_itr % 10 == 0:
            try:
                self.proc.kill()
            except:
                pass
            self.proc = subprocess.Popen(call,
                                         stdout=sys.stdout,
                                         stderr=sys.stderr,
                                         preexec_fn=os.setsid)

        # connect to an instance to receive initial state information
        print("waiting for response from client at port {}...".format(PORT))
        self.conn, address = self.s.accept()
        print('Connected by', address)

        # get initial state data
        data = None
        while not data:  # TODO(ak): check if I need to do this
            data = self.conn.recv(1024)
        data = data.decode('ascii').split(":")

        # reset the values for the leading vehicle
        self.vehicles["leader"]["pos"] = 40
        self.vehicles["leader"]["speed"] = 0
        self.vehicles["leader"]["accel"] = 0

        # reset the initial values for the rl vehicles
        for i, veh_id in enumerate(self.rl_ids):
            self.vehicles[veh_id]["pos"] = float(data[3*i])
            self.vehicles[veh_id]["speed"] = float(data[3*i+1])
            self.vehicles[veh_id]["accel"] = float(data[3*i+2])

        # reset the headways
        for i, veh_id in enumerate(self.rl_ids):
            self.vehicles[veh_id]["headway"] = \
                self.vehicles[self.ids[i]]["pos"] \
                - self.vehicles[self.ids[i+1]]["pos"] - VEHICLE_LENGTH

        for veh_id in self.rl_ids:
            # empty the acceleration queues for each rl vehicle
            self.action_queue[veh_id].clear()

            # add zero acceleration terms for the first time delayed steps
            for _ in range(int(self.action_delay/self.sim_step)):
                self.action_delay.append(0)

        # increment the rollout iteration number
        self.rollout_itr += 1

        return list(self._get_state().T)
