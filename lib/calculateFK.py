import numpy as np
from math import pi

def compute_DH_matrix(a, alpha, d, theta):
    A = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                  [0, np.sin(alpha), np.cos(alpha), d],
                  [0, 0, 0, 1]])
    return A

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        DH_param0 = {'a': 0, 'alpha': 0, 'd': 0.141, 'theta': 0}
        DH_param1 = {'a': 0, 'alpha': -pi/2, 'd': 0.192, 'theta': 0}
        DH_param2 = {'a': 0, 'alpha': pi/2, 'd': 0, 'theta': 0}
        DH_param3 = {'a': 0.0825, 'alpha': pi/2, 'd': 0.195+0.121, 'theta': 0}
        DH_param4 = {'a': 0.0825, 'alpha': pi/2, 'd': 0, 'theta': pi/2+pi/2}
        DH_param5 = {'a': 0, 'alpha': -pi/2, 'd': 0.125+0.259, 'theta': 0}
        DH_param6 = {'a': 0.088, 'alpha': pi/2, 'd': 0, 'theta': -pi/2-pi/2}
        DH_param7 = {'a': 0, 'alpha': 0, 'd': 0.051+0.159, 'theta': -pi/4}

        self.DH_params = [DH_param0, DH_param1, DH_param2, DH_param3, DH_param4, DH_param5, DH_param6, DH_param7]
    
    def compute_DH_i(self, i, q):
        # give the A matrix from link i-1 to i
        assert i >= 1
        id = i - 1
        DH_param = self.DH_params[id]
        a = DH_param['a']
        alpha = DH_param['alpha']
        d = DH_param['d']
        theta = DH_param['theta'] + q[id - 1] if id > 0 else DH_param['theta'] 
        return compute_DH_matrix(a, alpha, d, theta)

    def compute_Hi(self, i, q):
        # give the H matrix from base to link i
        assert i >= 1
        for k in range(1, i+1):
            if k == 1:
                H = self.compute_DH_i(k, q)
            else:
                H = H @ self.compute_DH_i(k, q)
        return H

    def forward_dbg(self, q):
        T0s = [self.compute_Hi(i, q) for i in range(1, 9)]
        T0s.insert(0, np.eye(4))
        return T0s

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here
        jointPosition1 = (self.compute_Hi(1, q) @ np.array([0, 0, 0, 1]))[:3]
        jointPosition2 = (self.compute_Hi(2, q) @ np.array([0, 0, 0, 1]))[:3]
        jointPosition3 = (self.compute_Hi(3, q) @ np.array([0, 0, 0.195, 1]))[:3]
        jointPosition4 = (self.compute_Hi(4, q) @ np.array([0, 0, 0, 1]))[:3]
        jointPosition5 = (self.compute_Hi(5, q) @ np.array([0, 0, 0.125, 1]))[:3]
        jointPosition6 = (self.compute_Hi(6, q) @ np.array([0, 0, -0.015, 1]))[:3]
        jointPosition7 = (self.compute_Hi(7, q) @ np.array([0, 0, 0.051, 1]))[:3]
        jointPosition8 = (self.compute_Hi(8, q) @ np.array([0, 0, 0, 1]))[:3]

        jointPositions = np.stack([jointPosition1, jointPosition2, jointPosition3, jointPosition4, jointPosition5, jointPosition6, jointPosition7, jointPosition8])
        assert jointPositions.shape == (8, 3)
        T0e = self.compute_Hi(8, q)

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        for i in range(1, 8):
            H = self.compute_Hi(i, q)
            if i == 1:
                axis_of_rotation_list = H[:3, 2]
            else:
                axis_of_rotation_list = np.vstack([axis_of_rotation_list, H[:3, 2]])
        
        axis_of_rotation_list = axis_of_rotation_list.T
        assert axis_of_rotation_list.shape == (3, 7)
        return(axis_of_rotation_list)

    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    # q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    q = np.array([0., 0., 0., 0., 0., 0., 0.])

    joint_positions, T0e = fk.forward(q)
    T0e = np.round(T0e, 3)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
