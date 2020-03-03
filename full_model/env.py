import mujoco_py
import random
import math
import time
import os
import numpy as np
import gym


class UR5Environment:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        mjb_bytestring = mujoco_py.load_model_from_path(xml_path).get_mjb()
        self.model = mujoco_py.load_model_from_mjb(mjb_bytestring)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)

    def step(self):
        self.sim.step()
        self.viewer.render()

    def setGoal(self, qpos):
        for i in range(len(qpos)):
            self.sim.data.ctrl[i] = qpos[i]

    def goalReached(self, qpos):
        for i in range(len(qpos)):
            if abs(self.sim.data.qpos[i] - qpos[i]) > 0.05:
                return False
        return True


if __name__ == '__main__':
    xml_path = os.path.dirname(os.path.realpath(__file__)) + '/UR5_2fgripper.xml'
    env = UR5Environment(xml_path)

    initialPos = [-math.pi / 2, -math.pi / 2, 
                   -math.pi / 2, -math.pi / 2,  math.pi / 2, 0]
    env.setGoal(initialPos)
    initPosReached = False

    try:
        while True:
            env.step()
            if not initPosReached and env.goalReached(initialPos):
                initPosReached = True
                print("Goal reached!")
    except KeyboardInterrupt:
        print('Shutdown environment!')
