import numpy as np

class ElectronPhotonSeparator():

    def __init__(self, **kwargs):
        # TODO: this class could accept some config parameters
        pass

    def likelihood_fractions(self, reco_showers, sh_energy_masked):
        '''
        Obtain the electron- and photon likelihood fractions for the showers by looking at the dE/dx value at the very start of an EM shower.
        Inputs:
            - reco_showers (M x 1): Array of M shower objects (defined in chain.py)
            - sh_energy_masked (N x 5): All energy deposits (x,y,z,batch_id,edep) which have semantic segmentation 'shower'
        Returns:
            - L_electron (M x 1): Array of M electron likelihood fractions (one likelihood fraction per showers)
            - L_photon (M x 1): Array of M photon likelihood fraction (one likelihood fraction per showers)
        '''

        #Moyal fitted parameters for electron induced showers:
        # TODO: ADJUST PARAMETERS DEPENDING ON THE SAMPLE (e.g. e_scale_up corresponds to the weighting factor)
        e_scale_up     = 1. #6089.115655656354
        e_shift_dEdx   = 9.709773057401973
        e_squeeze_dEdx = 4.961465355579281

        #Moyal fitted parameters for photon induced showers:
        # TODO: ADJUST PARAMETERS DEPENDING ON THE SAMPLE (e.g. p_scale_up corresponds to the weighting factor)
        p_scale_up     = 1. #4471.081346220595
        p_shift_dEdx   = 10.232749303857846
        p_squeeze_dEdx = 2.5419640824853302

        # Define the radius of the sphere in which the edeps are summed up, and the pixel pitch
        radius      = 7.  # TODO: Adjust parameter
        pixel_pitch = 0.3 # [cm]

        # Loop over all shower objects, sum up the edeps in the sphere and calculate the likelihood fractions
        for sh_index, sh in enumerate(reco_showers):
            #print(' sh.start:      ', sh.start)
            #print(' sh.direction:  ', sh.direction)
            #print(' sh.voxels:     ', sh.voxels)
            #print(' sh.energy:     ', sh.energy)
            #print(' sh.pid:        ', sh.pid)
            #print(' sh.group_pred: ', sh.group_pred)
            #print(' sh.L_e:        ', sh.L_e)
            #print(' sh.L_p:        ', sh.L_p)

            coords     = sh_energy_masked[sh.voxels][:,0:3] + 0.5 # add 0.5 in order to get the voxel middle
            edeps      = sh_energy_masked[sh.voxels][:,4]

            summed_edeps = 0.

            for edep_index, edep in enumerate(edeps):
                diff = np.linalg.norm(sh.start - coords[edep_index])

                if diff < radius:
                    #print(' coords: ', coords[edep_index], ' \t edep: ', edep)
                    summed_edeps += edep

            dEdx = summed_edeps/(radius*pixel_pitch)
            #print(' summed_edeps: ', summed_edeps)
            #print(' dE/dx: ', dEdx)

            # Obtain likelihoods
            e_likelihood = self.moyal(dEdx, e_scale_up, e_shift_dEdx, e_squeeze_dEdx)
            p_likelihood = self.moyal(dEdx, p_scale_up, p_shift_dEdx, p_squeeze_dEdx)
            #print(' e_likelihood: ', e_likelihood)
            #print(' p_likelihood: ', p_likelihood)

            # Obtain likelihood fractions
            sh.L_e = e_likelihood / (e_likelihood + p_likelihood)
            sh.L_p = p_likelihood / (e_likelihood + p_likelihood)

            if (sh.L_e + sh.L_p) < 0.999 or (sh.L_e + sh.L_p) > 1.001:
                print(' WARNING: L_tot = L_e + L_p =', sh.L_e + sh.L_p)

            '''
            if sh.L_e > sh.L_p:
                print(' L_e:   ', sh.L_e, ' <------ ELECTRON ')
                print(' L_p:   ', sh.L_p)
                print(' --- ')
            else:
                print(' L_e:   ', sh.L_e)
                print(' L_p:   ', sh.L_p, ' <------ PHOTON ')
                print(' --- ')
            '''

        return


    def moyal(self, dEdx, scale_up, shift_dEdx, squeeze_dEdx):
        return scale_up * 1./(np.sqrt(2.*np.pi)) * np.exp(-0.5*((squeeze_dEdx*dEdx-shift_dEdx)+np.exp(-(squeeze_dEdx*dEdx-shift_dEdx))))
