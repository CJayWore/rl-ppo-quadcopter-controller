# Membership functions and fuzzifiers
import numpy as np
import skfuzzy as skf

class GaussianMembership:
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def fuzzify(self, x):
        return skf.gaussmf(x, self.mean, self.stddev)
    
class Fuzzifier:
    def __init__(self, membership_functions):
        self.mfs = [
            [GaussianMembership(c, s) for c, s in mf_list]
            for mf_list in membership_functions
        ]

    def fuzzify(self, x):
        fuzzified = []
        for i, mf_list in enumerate(self.mfs):
            fuzzified.append([mf.compute(x[i]) for mf in mf_list])
        return np.array(fuzzified)  # shape: (n_inputs, n_mfs)