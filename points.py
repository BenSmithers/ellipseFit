"""
I'm keeping the point-getting code and putting it in here
"""

import os 
import json 
import numpy as np

dimdict = {
    'Amp0'   :     0,
    "Amp1"   :     1,
    "Amp2"   :     2,
    "Amp3"   :     3,
    "Amp4"   :     4,
    "Phs1"   :     5,
    "Phs2"   :     6,
    "Phs3"   :     7,
    "Phs4"   :     8
}

DIM  = 9

basis_vectors = np.diag(np.ones(9))


def get_ice_points():
    """
        This is used to load in all the points that lie on the ellipsoidal likelihood contour 

        It's a list of 9-dimensional points

        A `scale` term is used here to make things bigger, since it doesn't seem like the fitter really likes small things 
    """

    _obj = open(os.path.join(os.path.dirname(__file__),"all_widths.json"), 'rt')
    data = json.load(_obj)
    _obj.close()
    datapoints = []

    def get_width(param, mode):
        """
        function for getting the width of Amp/Phase "param" for mode `mode`

        note: these are in units of the parameters  
        """
        subdict = data[param]
        for entry in subdict:
            if str(mode).lower()==entry["mode0"]:
                return entry["width"]
        
        for entry in subdict:
            print(str(mode))

    

    params = ["Amp", "Phs"]
    modes = [0,1,2,3,4]
    all_widths = []
    for param in params:
        for mode in modes:
            if mode==0 and param =="Phs": # no phase zero 
                continue
            else:
                all_widths.append(get_width(param, mode))

    scale = 1/np.mean(all_widths)

    for subtype in data.keys():
        # we have Amp, Phs, and AmpPhs
        for entry in data[subtype]:
            # skip modes greater than 4 for now
            if int(entry["mode0"])>4 or int(entry["mode1"])>4:
                continue

            prim_key = entry["param0"] + entry["mode0"]
            sec_key = entry["param1"] + entry["mode1"]

            
            if entry["param0"] == entry["param1"]:
                # on-axis points 
                point = basis_vectors[dimdict[prim_key]]*(entry["width"])*scale
                datapoints.append(point)

                point = -1*basis_vectors[dimdict[prim_key]]*(entry["width"])*scale
                datapoints.append(point)
            else:
                # this is an amplitude/phase entry... so we need to go find scaling factors for both 
                
                # ie Amp1 Phase3
                basis_0_scale = get_width(entry["param0"], entry["mode0"])  
                basis_1_scale = get_width(entry["param1"], entry["mode1"])

                # off-axis points are in term of sigmas, so we scale these up 
                # one by Amp1 and the other by Phase3 
                point = basis_0_scale*basis_vectors[dimdict[prim_key]]*(entry["width"])*scale \
                            + basis_1_scale*basis_vectors[dimdict[sec_key]]*(entry["width"] )*scale
                datapoints.append(point)

                point = -1*basis_0_scale*basis_vectors[dimdict[prim_key]]*(entry["width"] )*scale \
                         - basis_1_scale*basis_vectors[dimdict[sec_key]]*(entry["width"])*scale
                #datapoints.append(point)
    return datapoints