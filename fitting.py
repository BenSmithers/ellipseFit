from rotations import construct_rotation
from points import get_ice_points

import numpy as np
from math import pi
import os 
import json 

from scipy.optimize import minimize


class EllipseFit:
    def __init__(self, point_list:list, use_previous_fit = True):
        self._n_dim = len(point_list[0])

        self._datapoints = point_list

        # the ellipse represents the 1sigma contour for the likelihood distribution 
        # this 50 comes from ~45 DOF 1sigma  - might not be right 
        self._ellipse_rad = 50

        self.bounds = []
        for i in range(self._n_dim): # 9
            self.bounds.append([1e-4, np.inf])
        for i in range(int(self._n_dim*(self._n_dim-1)/2)):  # 36 
            self.bounds.append([0, 4*pi])

        self.options={
            'maxiter':1e8,
            'maxfev':1e8,
            'maxfun':1e8,
            'ftol':1e-20,
            'gtol':1e-20,
            'eps':1e-20
        }

        self._state_file = os.path.join(os.path.dirname(__file__), "ellipse_state.json")

        self._count = 0
        self._use_previous = use_previous_fit

    def _minfunc(self, params):
        """
            params has nine entries for ellipse axis values 
            and then the other 36 are rotation matrix angles

            The fitter tries to find the rotation that puts all the points onto an ellipse whose axes go along the canonical X,Y,Z,W,W2,W3, etc axes
                while also trying to find the magnitude for the ellipse along each of those axes 
        """
        rotation_matrix = construct_rotation(thetas=params[self._n_dim:], dim=self._n_dim)
        sumLLH = 0.0

        # we now rotate the points we have according to the rotation matrix
        # if the rotation angle is just right, it will have the points all lie on the ellipse that goes along the axes 
        for point in self._datapoints:
            rotated_point = np.matmul(rotation_matrix, point)

            # right now, doing the simple least-squares method.
            # TODO maybe try cauchy thingy 
            sumLLH+= (np.sum((np.array(1./params[0:self._n_dim])*rotated_point**2)) - self._ellipse_rad)**2
        return sumLLH

    def callback(self, paramset):
        self._count += 1 
        if self._count%50==0:
            print(paramset)
        return False

    def minimize(self, n_iter = 1):
        """
            Load in the state file (if one was run earlier), perturb it, and run the fitter again
            This is done to hopefully basin-hop while still being in this minimum neighborhood 
        """
        for i in range(n_iter):
            if os.path.exists(self._state_file) and self._use_previous:
                _obj = open(self._state_file, 'rt')
                data  = json.load(_obj)
                _obj.close()
                x0 = data["params"]

                x0 = [(1+0.1*np.random.rand())*val for val in x0]

            else:
                pars = [10 for i in range(self._n_dim)]
                rots = (1.0 + 0.2*np.random.rand(int(self._n_dim*(self._n_dim-1)/2))).tolist()
                x0 = pars + rots

            result = minimize(self._minfunc, x0, bounds=self.bounds, callback = self.callback, options=self.options)
            
            print(result)

            params = result.x

            new_data = {
                "params": [param for param in params],
                "value":self._minfunc(result.x)
            }

            if not os.path.exists(self._state_file):
                _obj = open(self._state_file, 'wt')
                json.dump(new_data, _obj, indent=4)
                _obj.close()
            else:
                if (new_data["value"] < data["value"]) and result.success:
                    print("updating fit file!")

                    _obj = open(self._state_file, 'wt')
                    json.dump(new_data, _obj, indent=4)
                    _obj.close()
                else:
                    if not result.success:
                        print("Failure!")
                    else:
                        print("Worse fit... ")


if __name__=="__main__":
    fitter = EllipseFit(get_ice_points())
    fitter.minimize(10)