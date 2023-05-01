files =[
    "Amp.txt",
    "Amp0.txt",
    "Phs.txt",
    "AmpPhs.txt",
    "AmpAmp.txt",
    "PhsPhs.txt"
]

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

import numpy as np
import os 
from math import pi, sqrt

def extract(filename, amp_scales, pha_scales):
    """
        Takes the data in a file and parses it into nuisance-point v LLH 
    """
    rawdata = np.transpose(np.loadtxt(filename, dtype=str))

    paramset = rawdata[0]
    mode_str = rawdata[1]
    #mode_str = np.transpose([entry.split("_") for entry in rawdata[1]])
    all_modes = np.unique(mode_str)
    #mode_1 = mode_str[0].astype(int)
    #mode_2 = mode_str[1].astype(int)

    shift_vals = rawdata[2]
    llhs = rawdata[3]
    dom_no = rawdata[4]

    all_params = np.unique(paramset)
    scaleval = 59
    all_data = []
    for param in all_params:
        for mode_pair in all_modes:
            # we first seek out the lines specific for a given Parameter and Mode-set 
            mask = np.logical_and(param==paramset, mode_pair==mode_str)
            # here, so now we have a mask that gets all the LLH and shifts for a given param/mode combo 


            these_shifts = shift_vals[mask]
            unique_shifts =np.unique(these_shifts)

            
            # sum up the LLH per-DOM for each shift value 
            net_llhs = []
            for shift in unique_shifts:
                these_llhs = llhs[np.logical_and(mask, shift_vals==shift)]
                net_llhs.append(np.sum(these_llhs.astype(float)*scaleval/len(these_llhs.astype(float))) )

            # makes the fitting easier. 
            # we use the \Delta LLH metric 
            net_llhs = np.array(net_llhs) - min(net_llhs)

            orgvals = np.transpose([unique_shifts.astype(float), net_llhs])
            orgvals= np.array(list(sorted(orgvals, key=lambda entry:entry[0])))
            orgvals=np.transpose(orgvals)

            if "AmpPhs"==param:
                t0 = "Amp"
                t1 = "Phs"
            elif "Amp" in param:
                t0 = "Amp"
                t1 = "Amp"
            elif "Phs" in param:
                t0 = "Phs"
                t1 = "Phs"
            else:
                raise Exception(param)

            scale_factor = 1.0

            split = mode_pair.split("_")
            if len(split) == 1:
                mode0 = split[0]
                mode1 = split[0]
            else:
                mode0 = split[0]
                mode1= split[1]

                if t0=="Amp":
                    width0 = amp_scales[int(mode0)]
                else:
                    width0 = pha_scales[int(mode0)-1]
                if t1=="Amp":
                    width1 = amp_scales[int(mode1)]
                else:
                    width1 = pha_scales[int(mode1)- 1]
                scale_factor = sqrt(width0**2 + width1**2)

            orgvals[0]*=scale_factor

            mask = orgvals[1]<10

            parameter_values = orgvals[0][mask]
            llhs = orgvals[1][mask]

fn0 = os.path.join(os.path.dirname(__file__), "llh_values", files[2])
extract(fn0)