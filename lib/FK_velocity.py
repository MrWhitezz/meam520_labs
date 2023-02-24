import numpy as np 
from lib.calcJacobian import calcJacobian

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """

    J = calcJacobian(q_in)
    dp = dq.reshape((7,1))
    velocity = J @ dq

    print('velocity', velocity.shape)
    print('dq', dq.shape)
    print('J', J.shape)
    velocity = velocity.reshape((6,1))

    assert velocity.shape == (6, 1)
    
    return velocity


if __name__ == '__main__':
    q = np.array([0, 0, 0, 0, 0, 0, 0])
    dq = np.array([1, 0, 0, 0, 0, 0, 0])
    v = FK_velocity(q, dq)
    print('v', v)
    linear_velocity = v[:3]
    angular_velocity = v[3:]
    print('linear_velocity', linear_velocity)
    print('angular_velocity', angular_velocity)