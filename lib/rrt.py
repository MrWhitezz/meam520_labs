import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK
from lib.tree import TreeNode

def isRobotCollided(map, q):
    """
    :param map:         the map struct
    :param q:           the configuration of the robot
    :return:            returns True if the robot is in collision with the environment, False otherwise
    """
    obstacles = map.obstacles
    if len(obstacles) > 0:
        fk = FK()
        joint_positions, _ = fk.forward(q)
        for obs_idx in range(len(obstacles)):
            box = obstacles[obs_idx, :].copy()
            for i in range(0, 7):
                linePt1 = joint_positions[i, :].reshape(1, 3).copy()
                linePt2 = joint_positions[i+1, :].reshape(1, 3).copy()
                if True in detectCollision(linePt1, linePt2, box):
                    # print('linePt1', linePt1, 'linePt2', linePt2, 'box', box)
                    return True
    return False

def isPathCollided(map, q1, q2):
    """
    :param map:         the map struct
    :param q1:          the start configuration of the robot
    :param q2:          the end configuration of the robot
    :return:            returns True if the robot is in collision with the environment, False otherwise
    """
    step = 0.01
    dq = q2 - q1
    n_steps = int(np.linalg.norm(dq) / step)
    # print('n_steps', n_steps)
    for i in range(n_steps):
        q = q1 + i * step * dq
        if isRobotCollided(map, q):
            return True
    return False
        

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    # path = [start]
    np.random.seed(0)

    n_samples = 10000
    s = isRobotCollided(map, goal)
    t = isRobotCollided(map, start)
    if s or t:
        # print('start or goal is in collision')
        return np.array([])
    s_to_t = isPathCollided(map, start, goal)
    # print('s_to_t', s_to_t)

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    root = TreeNode(start)
    for k in range(n_samples):
        q_rand = np.random.uniform(low=lowerLim, high=upperLim)
        assert q_rand.shape == (7,)

        if isRobotCollided(map, q_rand):
            continue
            
        _, node_nearest = root.traverse_for_min_distance(q_rand)
        if isPathCollided(map, node_nearest.data, q_rand):
            continue
        node_rand = TreeNode(q_rand)
        node_nearest.add_child(node_rand)

        if isPathCollided(map, q_rand, goal):
            continue

        node_goal = TreeNode(goal)
        node_rand.add_child(node_goal)
        path = node_goal.root_path()
        return np.array(path)
            
    return np.array([])

if __name__ == '__main__':
    starts = [np.array([0, -1, 0, -2, 0, 1.57, 0]),
            np.array([-1.2, 0.4, 0.7, -1.5, -0.2, 1.8, 0.707])]
    goals = [np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7]),
            np.array([1.2, 0.4, -0.7, -1.5, 0.2, 1.8, 0.707])]
    map_struct = loadmap("../maps/map1.txt")
    start = starts[0]
    goal =  goals[0]
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    print('path', path)
    print('start', start)
    print('goal', goal)
