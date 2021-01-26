import numpy as np

class Shower:
    '''
    Data structure that contains all the reconstructed shower information
    '''
    def __init__(self, start=-np.ones(3), direction=-np.ones(3), voxels=[], energy=-1., pid=-1, L_e=-1., L_p=-1.):
        self.start      = start
        self.direction  = direction
        self.voxels     = voxels
        self.energy     = energy
        self.pid        = int(pid)
        self.L_e        = L_e # electron (positron) likelihood fraction
        self.L_p        = L_p # photon likelihood fraction

    def __str__(self):
        return """
        Shower ID  : {}
        Start point: ({:0.2f},{:0.2f},{:0.2f})
        Direction  : ({:0.2f},{:0.2f},{:0.2f})
        Voxel count: {}
        Energy     : {}
        Photon L   : {}
        Electron L : {}
        """.format(self.pid, *self.start, *self.direction, len(self.voxels), self.energy, self.L_p, self.L_e)
