import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import rq

import time


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                       or n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    
    if points_3d.shape[1] == 3:
        points_3d = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
   
    projected_points = P @ points_3d.T
    projected_points_2d = projected_points[:2, :] / projected_points[2, :] 
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return projected_points_2d.T


def objective_func(x, **kwargs):
    """
        Calculates the difference in image (pixel coordinates) and returns
        it as a 2*n_points vector

        Args:
        -        x: numpy array of 11 parameters of P in vector form
                    (remember you will have to fix P_34=1) to estimate the reprojection error
        - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                    retrieve these 2D and 3D points and then use them to compute
                    the reprojection error.
        Returns:
        -     diff: A 2*N_points-d vector (1-D numpy array) of differences between
                    projected and actual 2D points. (the difference between all the x
                    and all the y coordinates)

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

    points_2d = kwargs['pts2d'].flatten()
    points_3d = kwargs['pts3d']
    
    P = np.reshape(np.append(x, 1), (3,4))
    
    projected_points_2d = projection(P, points_3d).flatten()
    diff = points_2d - projected_points_2d
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return diff


def estimate_camera_matrix(pts2d: np.ndarray,
                           pts3d: np.ndarray,
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squres form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1)
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters.

              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.wjap

              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables
                          for the objective function
    '''

    start_time = time.time()

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

    x0 = initial_guess.flatten()[:-1]
    result = least_squares(
        objective_func, x0, method='lm', verbose=2, max_nfev=5000, kwargs={"pts2d":pts2d,"pts3d":pts3d})
    M = np.reshape(np.append(result.x, 1), (3, 4))
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    print("Time since optimization start", time.time() - start_time)

    return M

def decompose_camera_matrix(P: np.ndarray) -> (np.ndarray, np.ndarray):
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix

        Args:
        -  P: 3x4 numpy array projection matrix

        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

    K, R = rq(P[:, :3])
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
        K[:, -1] *= -1
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return K, R

def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray,
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (3,) representing the camera center
            location in world coordinates
    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

    M = P[:, :3]
    t = P[:, 3]

    cc = -np.linalg.inv(M) @ t
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return cc
