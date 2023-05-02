"""
This script loads all the ppc LLHs in directly and builds a list of points in 9d space; we save a log likelihood value for each of thoes sampled points 

Then, using the 45 free parameters for an arbitrary 9d covariance matrix, we create a lower-diagonal matrix
by operating that on its transpose, we can construct an arbitrary covariance matrix

Then we allow a fit to run on those 45 parameters, while comparing the LLHs we get from the fit cov mat to the ones we actually measured 
"""

import numpy as np
import os 
from math import pi, sqrt, log10
import json 
from scipy.linalg import expm, norm 
from scipy.optimize import minimize

FIT_FILE = os.path.join(os.path.dirname(__file__), "choleksky_fit.json")

_obj = open(os.path.join(os.path.dirname(__file__), "all_widths.json"),'rt')
width_dict = json.load(_obj)
_obj.close()

def get_shift(param0, param1, mode0, mode1):
    if mode0==mode1 and param0==param1:
        subdict = width_dict[param0]
    else:
        subdict = width_dict["AmpPhs"]

    for entry in subdict:
        if param0==entry["param0"] and param1==entry["param1"] or \
            param1==entry["param0"] and param0==entry["param1"] :
            if mode0==entry["mode0"] and mode1==entry["mode1"] or \
                mode1==entry["mode0"] and mode0==entry["mode1"]:
        
                return entry["center"]
    print(type(mode0))

    print(subdict)
    print(mode0 ,mode1, param0, param1)

    raise Exception()


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

unit_vectors = np.diag(np.ones(9))

rtwo = sqrt(2)

def get_width(param, mode):
    subdict = width_dict[param]
    for entry in subdict:
        if str(mode).lower()==entry["mode0"]:
            return entry["width"]
    
    for entry in subdict:
        print(str(mode))

param_types = ["Amp", "Phs"]
modes = [0,1,2,3,4]
all_widths = []
for param in param_types:
    for mode in modes:
        if param=="Phs" and mode==0:
            continue
        all_widths.append( get_width(param, mode) )
all_widths = np.array(all_widths)

scale_mat = np.zeros(shape=(9,9))
for i in range(9):
    for j in range(9):
        scale_mat[i][j] =all_widths[i]*all_widths[j]

def extract(filename, amp_scales, pha_scales):
    """
        Takes the data in a file and parses it into nuisance-point v LLHs

        We keep each sampled likelihood (nominalized over DOMs)
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
    all_points = []
    all_llhs = []
    for param in all_params:
        for mode_pair in all_modes:
            # we first seek out the lines specific for a given Parameter and Mode-set 
            mask = np.logical_and(param==paramset, mode_pair==mode_str)
            # here, so now we have a mask that gets all the LLH and shifts for a given param/mode combo 

            these_shifts  = shift_vals[mask]
            unique_shifts = np.unique(these_shifts)
            
            # sum up the LLH per-DOM for each shift value 
            net_llhs = []
            
            for shift in unique_shifts:
                these_llhs = llhs[np.logical_and(mask, shift_vals==shift)]
                net_llhs.append(np.sum(these_llhs.astype(float)*scaleval/len(these_llhs.astype(float))) ) # 

            # makes the fitting easier since we don't have to care about the prefactors (wooo)
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

            split = mode_pair.split("_")
            if len(split) == 1:
                mode0 = split[0]
                mode1 = split[0] # no, this is accurate... they're the same 
                rescale0 = 0.5
                rescale1 = 0.5
            else:
                mode0 = split[0]
                mode1= split[1]
                

                if t0=="Amp":
                    rescale0 = amp_scales[int(mode0)]
                else:
                    rescale0 = pha_scales[int(mode0)-1]
                if t1=="Amp":
                    rescale1 = amp_scales[int(mode1)]
                else:
                    rescale1 = pha_scales[int(mode1)-1]
                rescale0*=1/rtwo
                rescale1*=1/rtwo

            # large dLLH values can get weird
            mask = orgvals[1]<6

            parameter_values = orgvals[0][mask]
            cut_llhs = orgvals[1][mask]

            key0 = t0+mode0
            key1 = t1+mode1 

            reshift = get_shift(t0, t1, mode0, mode1)

            if int(mode0)>4 or int(mode1)>4:
                continue

            index0 = dimdict[key0]
            index1 = dimdict[key1]

            for i in range(len(cut_llhs)):
                all_points.append( (parameter_values[i]-reshift)*(rescale0*unit_vectors[index0] + rescale1*unit_vectors[index1]) )
                all_llhs.append( cut_llhs[i] ) 
    return all_points, all_llhs


def get_all_points()->'tuple[np.ndarray, np.ndarray]':
        
    files =[
        "Amp.txt",
        "Amp0.txt",
        "Phs.txt",
        "AmpPhs.txt",
        "AmpAmp.txt",
        "PhsPhs.txt"
    ]

    # we kinda cheat and load in the widths from the files first 
    import json
    _obj = open(os.path.join(os.path.dirname(__file__), "all_widths.json"), 'rt')
    data = json.load(_obj)
    _obj.close()

    amp_widths = [entry["width"] for entry in data["Amp"]]
    amp_widths = [amp_widths[-1],] + amp_widths[:-1]
    pha_widhts = [entry["width"] for entry in data["Phs"]]

    modes = [entry["mode0"] for entry in data["Amp"]]
    modes = [modes[-1],] + modes[:-1]

    all_points = []
    all_llhs = []

    for file in files:
        full_name = os.path.join(os.path.dirname(__file__), "llh_values", file)

        pts, llhs = extract(full_name, amp_widths, pha_widhts)
        all_points += pts
        all_llhs += llhs

    return np.array(all_points), np.array(all_llhs)


def build_cholesky_style(*params)->np.ndarray:
    """
        Look up the Cholesky decomposition! 
        For any matrix M, there exists a positive semi-definite matrix Sigma such that 
            Sigma = M @ M^T

        So we build a lower-diagonal matrix `ld`, and do that to it to build a positive semi-definite matrix

        I then divide that matrix by a thousand since scipy's minimizer works better with numbers O(10-100) than O(1e-4)
    """
    dim = 9 
    ld = np.zeros(shape=(dim, dim))
    ld[np.tril_indices(9,)] = params
    return np.dot(ld, ld.transpose())

count = 0
def fit_cov(nudge=False):
    """
        We wish to find the covariance matrix that assiciates the vectors with the likelhoods we get from the thing 
    """
    #from rotations import construct_rotation
    points, log_likelihoods = get_all_points()

    def callback(call):
        """
        Make sure it's not frozen...
        """
        global count 
        count +=1 
        if count%500 == 0:
            print("Working...")

    def minfunc(mat_params)->float:
        """
            TODO: need to optimize this stuff... 
            If there's a way to matrix multiply vectors of vectors I'd love to know it 
        """
        matrix= build_cholesky_style(*mat_params)
        n_points, dim = points.shape
        _points = points.reshape(n_points, 1, dim)

        result = np.matmul(np.matmul(_points, matrix), _points.transpose(0, 2, 1)).flatten()
        out = (0.5*result - log_likelihoods)**2
        return np.sum(out) 
    
    def nudge_func(mat_params)->float:
        """
            TODO: need to optimize this stuff... 
            If there's a way to matrix multiply vectors of vectors I'd love to know it 
        """
        matrix= build_cholesky_style(*mat_params)

        n_points, dim = points.shape
        _points = points.reshape(n_points, 1, dim)

        result = np.matmul(np.matmul(_points, matrix), _points.transpose(0, 2, 1)).flatten()
        out = np.log10( 1 + (0.5*result - log_likelihoods)**2)
        return np.sum(out )
    
    if False : # os.path.exists(FIT_FILE):
        print("Loading old fit")
        _obj = open(FIT_FILE, 'rt')
        data = json.load(_obj)
        _obj.close()

        x0 = np.array(data["fit"])
        print("Prefit value {}".format(data["llh"]))
        prefit_val = data["llh"]
        x0*= 1.0 + (0.01 if nudge else 0.2)*np.random.randn(len(x0))

    else:
        prefit_val = np.inf
        x0 = 1.0 + np.random.randn(45)*2.0
    
    bounds = [(-np.inf, np.inf) for i in range(45)]

    options={
            'maxiter':1e8,
            'maxfun':1e8,
            'ftol':1e-20,
            'gtol':1e-20,
            'eps':1e-20
        }
    
    result = minimize(nudge_func if nudge else minfunc, x0,bounds=bounds, options=options, callback=callback)

    value = minfunc(result.x)
    if value<prefit_val:
        print("Better fit! {} ".format(value))
        _obj = open(FIT_FILE,'wt')
        json.dump({
            "fit":result.x.tolist(),
            "llh":value
        },_obj, indent=4)
        _obj.close()
        
        
        #hess = build_cholesky_style(result.x)
        
        cor = construct_from_params(result.x)
        print(cor)

        keylist = list(dimdict.keys())
        outdict = {}
        # now we build the dictionary (yay)
        for i in range(9):
            outdict[keylist[i].lower()] = {}
            for j in range(9):
                outdict[keylist[i].lower()][keylist[j].lower()] = cor[i][j]

        print("Correlation matrix determinant {}".format(np.linalg.det(cor)))

        _obj= open(os.path.join(os.path.dirname(__file__), "ice_covariance.json"), 'wt')
        json.dump(outdict, _obj, indent=4)
        _obj.close()

        make_plot_for_params(result.x)

        return True

    print("Worse plot...")
    print(result)

    return False

def construct_from_params(params):
    hess =  build_cholesky_style(*params)

    nuisance_hessian = np.zeros_like(hess)
    for i in range(9):
        for j in range(9):
            nuisance_hessian[i][j] = (hess[i][j])/sqrt(hess[i][i]*hess[j][j])

    sad_inv = np.linalg.inv(nuisance_hessian)

    cor = np.ones_like(sad_inv)
    for i in range(9):
        for j in range(9):
            cor[i][j] = sad_inv[i][j]/sqrt(sad_inv[i][i]*sad_inv[j][j])

    return nuisance_hessian 


def load_res():
    """
        Makes a plot from a saved fit state
    """
    _obj = open(FIT_FILE, 'rt')
    data = json.load(_obj)
    _obj.close()

    x0 = np.array(data["fit"])
    make_plot_for_params(x0)

def make_plot_for_params(fitpar):
    """
        When provided the fit parameters, maeks a plot
    """
    import matplotlib.pyplot as plt 

    cor = construct_from_params(fitpar)

    fig, axes = plt.subplots()
    mesh = axes.pcolormesh(range(9), range(9), cor, vmin=-1, vmax=1, cmap='RdBu')
    axes.set_xticks(range(9), labels=dimdict.keys(), rotation=45)
    axes.set_yticks(range(9), labels=dimdict.keys())
    axes.hlines(y=4.5, xmin=-0.5, xmax=8.5, colors='k', linestyles='--')
    axes.vlines(x=4.5, ymin=-0.5, ymax=8.5, colors='k', linestyles='--')
    plt.colorbar(mesh)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),"plots", "choleskycov.png"),dpi=400)
    plt.show()

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--nudge", action="store_true", default=False,
                help="Use after running a fit. Uses a cauchy thing to relax penalties from far-off points.")
    options = parser.parse_args()
    nudge= options.nudge 

    print("Nudge mode: {}".format(nudge))

    while not fit_cov(nudge):
        continue
        

#test()
#load_res()