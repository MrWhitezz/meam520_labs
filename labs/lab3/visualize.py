import sys
from math import pi, sin, cos
import numpy as np
from time import perf_counter

import rospy
import roslib
import tf
import geometry_msgs.msg

from core.interfaces import ArmController

from lib.solveIK import IK

rospy.init_node("visualizer")

# Using your solution code
ik = IK()

#########################
##  RViz Communication ##
#########################

tf_broad  = tf.TransformBroadcaster()

# Broadcasts a frame using the transform from given frame to world frame
def show_pose(H,frame):
    tf_broad.sendTransform(
        tf.transformations.translation_from_matrix(H),
        tf.transformations.quaternion_from_matrix(H),
        rospy.Time.now(),
        frame,
        "world"
    )

#############################
##  Transformation Helpers ##
#############################

def trans(d):
    """
    Compute pure translation homogenous transformation
    """
    return np.array([
        [ 1, 0, 0, d[0] ],
        [ 0, 1, 0, d[1] ],
        [ 0, 0, 1, d[2] ],
        [ 0, 0, 0, 1    ],
    ])

def roll(a):
    """
    Compute homogenous transformation for rotation around x axis by angle a
    """
    return np.array([
        [ 1,     0,       0,  0 ],
        [ 0, cos(a), -sin(a), 0 ],
        [ 0, sin(a),  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def pitch(a):
    """
    Compute homogenous transformation for rotation around y axis by angle a
    """
    return np.array([
        [ cos(a), 0, -sin(a), 0 ],
        [      0, 1,       0, 0 ],
        [ sin(a), 0,  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def yaw(a):
    """
    Compute homogenous transformation for rotation around z axis by angle a
    """
    return np.array([
        [ cos(a), -sin(a), 0, 0 ],
        [ sin(a),  cos(a), 0, 0 ],
        [      0,       0, 1, 0 ],
        [      0,       0, 0, 1 ],
    ])

def transform(d,rpy):
    """
    Helper function to compute a homogenous transform of a translation by d and
    rotation corresponding to roll-pitch-yaw euler angles
    """
    return trans(d) @ roll(rpy[0]) @ pitch(rpy[1]) @ yaw(rpy[2])

#################
##  IK Targets ##
#################

# TODO: Try testing your own targets!

# Note: below we are using some helper functions which make it easier to generate
# valid transformation matrices from a translation vector and Euler angles, or a
# sequence of successive rotations around z, y, and x. You are free to use these
# to generate your own tests, or directly write out transforms you wish to test.

targets = [
    # transform( np.array([-.2, -.3, .5]), np.array([0,pi,pi])            ),
    # transform( np.array([-.2, .3, .5]),  np.array([pi/6,5/6*pi,7/6*pi]) ),
    # transform( np.array([.5, 0, .5]),    np.array([0,pi,pi])            ),
    # transform( np.array([.7, 0, .5]),    np.array([0,pi,pi])            ),
    # transform( np.array([.2, .6, 0.5]),  np.array([0,pi,pi])            ),
    # transform( np.array([.2, .6, 0.5]),  np.array([0,pi,pi-pi/2])       ),
    # transform( np.array([.2, -.6, 0.5]), np.array([0,pi-pi/2,pi])       ),
    # transform( np.array([.2, -.6, 0.5]), np.array([pi/4,pi-pi/2,pi])    ),
    # transform( np.array([.5, 0, 0.2]),   np.array([0,pi-pi/2,pi])       ),
    # transform( np.array([.4, 0, 0.2]),   np.array([pi/2,pi-pi/2,pi])    ),
    # transform( np.array([.4, 0, 0]),     np.array([pi/2,pi-pi/2,pi])    ),

    transform( np.array([.2, .3, 0.5]),  np.array([pi/4,pi/5,pi/2])  ),
    transform( np.array([-.2,.2, 0.6]),  np.array([pi/4,pi/2,pi/2])  ),
    transform( np.array([0., 0.2, 0.5]), np.array([pi/2,pi/2,pi/6])  ),
    transform( np.array([.5, -.1, 0.3]), np.array([0,pi/4,pi/2])  ),
    transform( np.array([.3, -.2, 0.4]), np.array([pi/2,0,pi/4])  ),
    transform( np.array([.3, 0, 0.4]),   np.array([pi/4, 0,pi/4])  ),
    transform( np.array([-.3, .5, 0.5]), np.array([pi/2, pi/2,pi])  ),
    transform( np.array([.5, .4, 0.5]),  np.array([pi/2, pi,pi/2])  ),
    transform( np.array([.7, -.1, 0.4]), np.array([pi/6, pi,pi])  ),
    transform( np.array([.2, -.3, 0.7]), np.array([pi/4, pi, pi])  ),
]

####################
## Test Execution ##
####################

np.set_printoptions(suppress=True)
dts = []
successes = []
perf_only = False

if __name__ == "__main__":

    arm = ArmController()
    args = sys.argv
    if len(args) > 1:
        assert args[1].startswith("p"), "Invalid argument"
        perf_only = True

    # Iterates through the given targets, using your IK solution
    # Try editing the targets list above to do more testing!
    for i, target in enumerate(targets):
        print("Target " + str(i) + " located at:")
        print(target)
        print("Solving... ")
        show_pose(target,"target")

        seed = arm.neutral_position() # use neutral configuration as seed

        start = perf_counter()
        q, success, rollout = ik.inverse(target, seed)
        stop = perf_counter()
        dt = stop - start

        dts.append(dt)
        successes.append(int(success))

        if success:
            print("Solution found in {time:2.2f} seconds ({it} iterations).".format(time=dt,it=len(rollout)))
            if perf_only:
                continue
            arm.safe_move_to_position(q)
        else:
            print('IK Failed for this target using this seed.')


        if i < len(targets) - 1:
            kk = input("Press Enter to move to next target...")
            if kk.startswith("k"):
                exit()

    dts = np.array(dts)
    mean_dt = np.mean(dts)
    median_dt = np.median(dts)
    max_dt = np.max(dts)
    min_dt = np.min(dts)
    print("Mean time: {time:2.2f} seconds".format(time=mean_dt))
    print("Median time: {time:2.2f} seconds".format(time=median_dt))
    print("Max time: {time:2.2f} seconds".format(time=max_dt))
    print("Min time: {time:2.2f} seconds".format(time=min_dt))
    print("Success rate: {rate:2.2f}%".format(rate=100*np.mean(successes)))
