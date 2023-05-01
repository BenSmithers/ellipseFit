from math import pi 
import numpy as np
import os 
from fitting import EllipseFit
from rotations import construct_rotation



def generate_ellipse_points(dim:int):
    n_theta = int(dim*(dim-1)/2)
    
    radius = 1.
    scales = (np.random.rand(dim)*radius + 1)
    angles = np.random.rand(n_theta)*2*pi

    params = np.concatenate((1./scales, angles))

    rotation = construct_rotation(angles, dim)
    rotation = np.diag(np.ones(dim))

    # use the scales to get x,y,z points
    
    n_points = dim + 3*dim*dim

    # xs, ys, zs, minus the last dim
    
    points_m1 = np.transpose([np.random.rand(n_points)*(radius/scales[0])-0.4*radius for i in range(dim-1)])

    pminus = np.random.randint(0,2, n_points)*2 -1 
    z_values = pminus*np.sqrt((radius**2 -  np.sum((1./scales[:-1])*points_m1**2, axis=1))*scales[-1])

    pointlist = []
    for i in range(n_points):
        pointlist.append(np.append(points_m1[i], z_values[i]))

    rot_points = np.array([np.matmul(rotation, point) for point in pointlist])

    print(np.shape(pointlist))
    return rot_points, params


statefile = os.path.join(os.path.dirname(__file__), "testing", "statefile.json")
if os.path.exists(statefile):
    os.remove(statefile)

rot_points, params = generate_ellipse_points(2)

import matplotlib.pyplot as plt
for point in rot_points:
    plt.plot(point[0], point[1], 'rd', ls='')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

new_fit = EllipseFit(rot_points,minfunc='least_sq', statefile=statefile)
fitresult = new_fit.minimize(50)

def plot_ellipse(params):
    min_x = -1.5# (params[0])-0.2
    max_x = 1.5 #(params[0])+0.2
    xs = np.linspace(min_x, max_x, 100)
    ys = np.sqrt(1.0 -  (1./params[0])*xs**2 )*params[1]

    rotation = construct_rotation(params[-1:], 2)

    all_pt = np.transpose([xs,ys])
    rot_pot = np.transpose(np.array([np.matmul(rotation, point) for point in all_pt]))


    plt.plot(rot_pot[0], rot_pot[1], color='blue')
    plt.show()
plot_ellipse(fitresult) 
plt.show()


print(fitresult)
print(params)

