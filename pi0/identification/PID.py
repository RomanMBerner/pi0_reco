import numpy as np
from scipy.spatial.distance import cdist

class ShowerIdentifier():

    def __init__(self, **cfg):
        '''
        Extracts the shower idenfication parameters from the module configuration
        '''
        self._max_distance  = cfg.get('max_distance', 7)     # Maximum distance from the shower start over which to integrate the energy
        self._voxel_size    = cfg.get('voxel_size', 0.3)     # Image voxel size in cm

        self._e_scale       = cfg.get('e_scale', 1.)         # Prior knowledge of the electron fraction in the sample of showers
        self._e_shift       = cfg.get('e_shift', 9.710)      # Peak of the electron dEdx distribution
        self._e_squeeze     = cfg.get('e_squeeze', 4.961)    # Inverse width (?) of the electron dEdx distribution

        self._p_scale       = cfg.get('p_scale', 1.)         # Prior knowledge of the electron fraction in the sample of showers
        self._p_shift       = cfg.get('p_shift', 10.233)     # Peak of the electron dEdx distribution
        self._p_squeeze     = cfg.get('p_squeeze', 2.542)    # Inverse width (?) of the electron dEdx distribution

        self._min_sep       = cfg.get('min_sep', 3)          # Minimal separation between shower start and vertex to be considered a photon

    def moyal(self, dEdx, type):
        '''
        Computes the value of the Moyal distribution (approx. to Landau) at a specific dEdx

        Inputs:
            - dEdx: Energy deposition per unit length in MeV/cm
            - type: Particle type ('p' for photon, 'e' for electron)
        Returns:
            - Value of the Moyal distribution in dEdx
        '''
        a, b, c = getattr(self, f'_{type}_scale'), getattr(self, f'_{type}_shift'), getattr(self, f'_{type}_squeeze')
        return a * 1./(np.sqrt(2.*np.pi)) * np.exp(-0.5*((c*dEdx-b)+np.exp(-(c*dEdx-b))))

    def set_edep_lr(self, showers, energy):
        '''
        Sets the electron and photon likelihood ratios for the showers by looking at
        the dE/dx value at the very start of an EM shower.

        Inputs:
            - showers (M x 1): Array of M shower objects (defined in chain.py)
            - energy (N x 5): All energy deposits (x,y,z,batch_id,edep) which have semantic segmentation 'shower'
        '''

        # Loop over all shower objects, sum up the energy depositions within some distance
        # and calculate the likelihood fractions from the dEdx distributions.
        for sh_index, sh in enumerate(showers):

            # Find distance from start to all other shower voxels
            coords = energy[sh.voxels,:3]
            dists  = cdist(sh.start.reshape(1,3), coords).flatten()

            # Sum all energies within the maximum allowed distance, convert to dEdx
            edeps      = energy[sh.voxels, -1]
            total_edep = np.sum(edeps[dists < self._max_distance])
            dEdx       = total_edep / (self._max_distance * self._voxel_size)

            # Obtain likelihoods
            e_likelihood = self.moyal(dEdx, 'e')
            p_likelihood = self.moyal(dEdx, 'p')

            # Obtain likelihood fractions
            if (e_likelihood + p_likelihood) > 0:
                sh.L_e = e_likelihood / (e_likelihood + p_likelihood)
                sh.L_p = p_likelihood / (e_likelihood + p_likelihood)
            else:
                sh.L_e = float(dEdx < self._e_shift/self._e_squeeze)
                sh.L_p = float(dEdx > self._p_shift/self._p_squeeze)

    def set_vertex_lr(self, showers, vertices):
        '''
        Sets the electron and photon likelihood ratios for the showers by looking at
        their distance from the closest interaction vertex.

        Inputs:
            - showers (M x 1): Array of M shower objects (defined in chain.py)
            - vertices (N x 3): List of potential interaction vertices
        '''

        # Loop over all shower objects, sum up the energy depositions within some distance
        # and calculate the likelihood fractions from the dEdx distributions.
        for sh_index, sh in enumerate(showers):

            # Find how close the shower start is from the closest vertex candidate
            dists = cdist(sh.start.reshape(1,3), vertices)
            sep   = np.min(dists)
            print(sep)

            # Give a photon an LR of 1 if the shower is removed by more than min_sep, 0 otherwise
            sh.L_e = float(sep <= self._min_sep)
            sh.L_p = float(sep >  self._min_sep)
