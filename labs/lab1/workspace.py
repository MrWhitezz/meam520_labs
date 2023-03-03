from lib.calculateFK import FK
from core.interfaces import ArmController

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fk = FK()

# the dictionary below contains the data returned by calling arm.joint_limits()
limits = [
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -1.7628, 'upper': 1.7628},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -3.0718, 'upper': -0.0698},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -0.0175, 'upper': 3.7525},
    {'lower': -2.8973, 'upper': 2.8973}
 ]

# TODO: create plot(s) which visualize the reachable workspace of the Panda arm,
# accounting for the joint limits.
#
# We've included some very basic plotting commands below, but you can find
# more functionality at https://matplotlib.org/stable/index.html

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def get_random_joint_angles():
    return np.array([np.random.uniform(limits[i]['lower'], limits[i]['upper']) for i in range(7)])

# TODO: update this with real results
# ax.scatter(1,1,1) # plot the point (1,1,1)
for i in range(1000):
    q = get_random_joint_angles()
    joints, T0e = fk.forward(q)
    ax.scatter(T0e[0,3], T0e[1,3], T0e[2,3])



plt.show()
