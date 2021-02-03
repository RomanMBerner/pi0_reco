import math
import numpy as np


# Class that contains all the reco shower information
class Shower():
    def __init__(self, start=-np.ones(3), direction=-np.ones(3), voxels=[], energy=-1., pid=-1, shower_group_pred=-1, L_e=-1., L_p=-1.):
        self.start             = start
        self.direction         = direction
        self.voxels            = voxels
        self.energy            = energy
        self.pid               = int(pid)
        self.shower_group_pred = int(shower_group_pred)
        self.L_e               = L_e # electron (positron) likelihood fraction
        self.L_p               = L_p # photon likelihood fraction
        
    def __str__(self):
        return """
        Shower ID       : {}
        Start point     : ({:0.2f},{:0.2f},{:0.2f})
        Direction       : ({:0.2f},{:0.2f},{:0.2f})
        Voxel count     : {}
        Energy          : {}
        Shower_group_pred : {}
        """.format(self.pid, *self.start, *self.direction, len(self.voxels), self.energy, self.shower_group_pred)

    
class ElectronShower(): # Including positron induced showers
    def __init__(self,
                 pid=-1,
                 group_id=-1,
                 start=-np.ones(4),
                 first_step=-np.ones(4),
                 first_edep=-np.ones(4),
                 mom_init=-np.ones(3),
                 direction=-np.ones(3),
                 voxels=[],
                 einit=-1,
                 edeps=[],
                 edep_tot=-1):
        self.pid        = int(pid)
        self.group_id   = int(group_id)
        self.start      = start      # creation vertex position (x,y,z,t)
        self.first_step = first_step # first step (x,y,z,t)
        self.first_edep = first_edep # first (in time) edep position (x,y,z,t)
        self.mom_init   = mom_init
        self.direction  = direction
        self.voxels     = voxels
        self.n_voxels   = len(self.voxels)
        self.einit      = einit
        self.edeps      = edeps
        self.edep_tot   = edep_tot

    def __str__(self):
        return """
        Shower ID  : {}
        Group ID   : {}
        Start      : ({:0.2f},{:0.2f},{:0.2f},{:0.2f})
        1st step   : ({:0.2f},{:0.2f},{:0.2f},{:0.2f})
        1st edep   : ({:0.2f},{:0.2f},{:0.2f},{:0.2f})
        Mom_init   : ({:0.2f},{:0.2f},{:0.2f})
        Direction  : ({:0.2f},{:0.2f},{:0.2f})
        Voxels     : {}
        Voxel count: {}
        E init     : {}
        E deposits : {}
        E dep tot  : {}
        """.format(self.pid, self.group_id, *self.start, *self.first_step, *self.first_edep, *self.mom_init, *self.direction, self.voxels, self.n_voxels, self.einit, self.edeps, self.edep_tot)


class PhotonShower(): # Including photons which make Compton Scattering
    def __init__(self,
                 pid=-1,
                 group_id=-1,
                 start=-np.ones(4),
                 first_step=-np.ones(4),
                 first_edep=-np.ones(4),
                 mom_init=-np.ones(3),
                 direction=-np.ones(3),
                 voxels=[],
                 einit=-1,
                 edeps=[],
                 edep_tot=-1):
        self.pid        = int(pid)
        self.group_id   = int(group_id)
        self.start      = start      # creation vertex position (x,y,z,t)
        self.first_step = first_step # first step (x,y,z,t)
        self.first_edep = first_edep # first (in time) edep position (x,y,z,t)
        self.mom_init   = mom_init
        self.direction  = direction
        self.voxels     = voxels
        self.n_voxels   = len(self.voxels)
        self.einit      = einit
        self.edeps      = edeps
        self.edep_tot   = edep_tot

    def __str__(self):
        return """
        Shower ID  : {}
        Group ID   : {}
        Start      : ({:0.2f},{:0.2f},{:0.2f},{:0.2f})
        1st step   : ({:0.2f},{:0.2f},{:0.2f},{:0.2f})
        1st edep   : ({:0.2f},{:0.2f},{:0.2f},{:0.2f})
        Mom_init   : ({:0.2f},{:0.2f},{:0.2f})
        Direction  : ({:0.2f},{:0.2f},{:0.2f})
        Voxels     : {}
        Voxel count: {}
        E init     : {}
        E deposits : {}
        E dep tot  : {}
        """.format(self.pid, self.group_id, *self.start, *self.first_step, *self.first_edep, *self.first_step, *self.first_edep, *self.mom_init, *self.direction, self.voxels, self.n_voxels, self.einit, self.edeps, self.edep_tot)


class ComptonShower(): # Only photon induced showers which make Compton Scattering
    def __init__(self,
                 pid=-1,
                 group_id=-1,
                 start=-np.ones(4),
                 first_step=-np.ones(4),
                 first_edep=-np.ones(4),
                 mom_init=-np.ones(3),
                 direction=-np.ones(3),
                 voxels=[],
                 einit=-1,
                 edeps=[],
                 edep_tot=-1):
        self.pid        = int(pid)
        self.group_id   = int(group_id)
        self.start      = start      # creation vertex position (x,y,z,t)
        self.first_step = first_step # first step (x,y,z,t)
        self.first_edep = first_edep # first (in time) edep position (x,y,z,t)
        self.mom_init   = mom_init
        self.direction  = direction
        self.voxels     = voxels
        self.n_voxels   = len(self.voxels)
        self.einit      = einit
        self.edeps      = edeps
        self.edep_tot   = edep_tot

    def __str__(self):
        return """
        Shower ID  : {}
        Group ID   : {}
        Start      : ({:0.2f},{:0.2f},{:0.2f},{:0.2f})
        1st step   : ({:0.2f},{:0.2f},{:0.2f},{:0.2f})
        1st edep   : ({:0.2f},{:0.2f},{:0.2f},{:0.2f})
        Mom_init   : ({:0.2f},{:0.2f},{:0.2f})
        Direction  : ({:0.2f},{:0.2f},{:0.2f})
        Voxels     : {}
        Voxel count: {}
        E init     : {}
        E deposits : {}
        E dep tot  : {}
        """.format(self.pid, self.group_id, *self.start, *self.first_step, *self.first_edep, *self.mom_init, *self.direction, self.voxels, self.n_voxels, self.einit, self.edeps, self.edep_tot)


class Analyser():

    def __init__(self):
        self.initialize_true()
        self.initialize_reco()


    def record(self, event, output, fiducial=0):
        self.extract_true_information(event, fiducial)
        self.find_true_electron_showers(event, output)
        self.find_true_photon_showers(event, output)
        self.extract_reco_information(event, output)

    def initialize_true(self):
        self.true_info = {}

        self.true_info['ev_id']                          = -1 # [-]
        self.true_info['n_pi0']                          = 0  # [-]
        self.true_info['n_gammas']                       = 0  # [-]
        self.true_info['pi0_track_ids']                  = [] # [-]
        self.true_info['gamma_group_ids']                = [] # [-]
        self.true_info['shower_particle_ids']            = [] # [-]
        #self.true_info['shower_particle_tids']          = [] # [-]
        #self.true_info['gamma_ids_making_compton_scat'] = [] # [-]      # List of photon ids which make compton scattering
        self.true_info['pi0_ekin']                       = [] # [MeV]
        self.true_info['pi0_mass']                       = [] # [MeV/c2] # Calculated with energy of the gammas and their momenta (invariant mass = sqrt(Etot-ptot))
        self.true_info['gamma_pos']                      = [] # [x,y,z]  # pi0 -> gamma+gamma vertex
        self.true_info['gamma_dir']                      = [] # [x,y,z]
        self.true_info['gamma_mom']                      = [] # [MeV/c]
        self.true_info['gamma_ekin']                     = [] # [MeV]    # initial energy of photon, = np.sqrt(p.px()**2+p.py()**2+p.pz()**2)
        self.true_info['gamma_edep']                     = [] # [MeV]
        #self.true_info['gamma_voxels']                  = [] # [-]      # List lists (for every shower 1 list) of voxels containing edeps
        self.true_info['gamma_n_voxels']                 = [] # [-]      # Number of voxels containing edeps for each shower
        self.true_info['gamma_first_step']               = [] # [x,y,z]  # Pos of the photon's 1st energy deposition
        #self.true_info['compton_electron_first_step']   = [] # [x,y,z]  # Pos of the compton electron's 1st energy deposition
        self.true_info['shower_first_edep']              = [] # [x,y,z]  # Pos of a shower's 1st (in time) energy deposition
        self.true_info['gamma_angle']                    = [] # [rad]    # Opening angle between the two photons from a true pi0 decay
        self.true_info['OOFV']                           = [] # [-]      # If shower has >0 edep(s) close to the LAr volume boundary.
                                                                         # Note: If photon leaves detector without producing an edep, this is NOT classified as OOFV.
                                                                         # For those events: Consider self.true_info['n_voxels']
        self.true_info['primaries_pdg_code']             = [] # [-]      # PDG codes of primary particles
        self.true_info['primaries_einit']                = [] # [MeV]    # initial energy of primary particles
        self.true_info['primaries_mom']                  = [] # [MeV/c]  # initial three momentum of primary particles
        return


    def initialize_reco(self):
        self.reco_info = {}

        self.reco_info['ev_id']                          = -1 # [-]
        self.reco_info['n_pi0']                          = 0  # [-]
        self.reco_info['n_gammas']                       = 0  # [-]
        self.reco_info['matches']                        = [] # [-]      # pairs of gamma indices for reconstructed pi0s
        self.reco_info['gamma_mom']                      = [] # [MeV/c]
        self.reco_info['gamma_dir']                      = [] # [x,y,z]
        self.reco_info['gamma_start']                    = [] # [x,y,z]  # pi0 -> gamma+gamma vertex
        self.reco_info['gamma_edep']                     = [] # [MeV]
        self.reco_info['gamma_pid']                      = [] # [-]
        self.reco_info['gamma_voxels_mask']              = [] # [-]
        self.reco_info['gamma_n_voxels_mask']            = [] # [-]
        self.reco_info['gamma_voxels']                   = [] # [-]
        self.reco_info['gamma_n_voxels']                 = [] # [-]
        self.reco_info['gamma_angle']                    = [] # [rad]    # Opening angle between the two photons from a reco pi0 decay
        self.reco_info['pi0_mass']                       = [] # [MeV/c2] # Reconstructed mass of the reco pi0
        self.reco_info['OOFV']                           = [] # [-]      # If shower has >0 edep(s) close to the LAr volume boundary.
                                                                         # Note: If photon leaves detector without producing an edep, this is NOT classified as OOFV.
                                                                         # For those events: Consider self.true_info['n_voxels']
        return


    def extract_true_information(self, event, fiducial):
        '''
        Obtain true informations about pi0s and gammas originated from pi0 decays and dump
        it to self.true_info['<variable>']
        '''

        #print(' self.true_info[ev_id]: ', self.true_info['ev_id'])
        self.true_info['ev_id'] = event['index']

        # Define some lists for pi0s, daughter photons and their compton electrons
        ids_of_true_photons           = [] # every photon has its own id
        tids_of_true_photons          = [] # every photon has its own tid
        parent_tids_of_true_photons   = [] # photons from the same pi0 decay have the same parent_track_id
        ids_of_photons_making_compton = [] # True or False for every true photon
        compton_electron_first_step   = [] # [x,y,z] coordinates of compton electrons first step
        
        # Get primary particles:
        # Primary pi0s are not in here...
        for i, p in enumerate(self.event['particles'][0]):
            #print(p.dump())
            if p.parent_pdg_code() == 0:
                self.true_info['primaries_pdg_code'].append(p.pdg_code())
                self.true_info['primaries_einit'].append(p.energy_init())
                self.true_info['primaries_mom'].append([p.px(),p.py(),p.pz()])
        
        # Get photons from pi0 decays:
        # Note: Photons from the same pi0 have the same parent_track_id.
        #       Every photon (from pi0 decay) has its own id.
        #       Primary pi0s do not have id and track_id
        for i, p in enumerate(event['particles']):
            #print(p.dump())
            if p.parent_pdg_code() == 111 and p.pdg_code() == 22:
                ids_of_true_photons.append(p.id())
                tids_of_true_photons.append(p.track_id())
                parent_tids_of_true_photons.append(p.parent_track_id())
                self.true_info['pi0_track_ids'].append(p.parent_track_id())
                self.true_info['gamma_pos'].append([p.x(),p.y(),p.z()])
                self.true_info['gamma_mom'].append([p.px(),p.py(),p.pz()])
                self.true_info['gamma_dir'].append([p.px(),p.py(),p.pz()]/np.linalg.norm([p.px(),p.py(),p.pz()]))
                self.true_info['gamma_ekin'].append(p.energy_init())
                self.true_info['gamma_first_step'].append([p.first_step().x(),p.first_step().y(),p.first_step().z()]) #, p.first_step().t()])
        self.true_info['n_pi0']    = len(np.unique(parent_tids_of_true_photons))
        self.true_info['n_gammas'] = len(ids_of_true_photons)

        # Obtain pi0 kinematic variable
        used_tids = []
        if self.true_info['n_pi0'] > 0:
            for i, p in enumerate(event['particles']):
                for r, idx1 in enumerate(parent_tids_of_true_photons):
                    for s, idx2 in enumerate(parent_tids_of_true_photons):
                        if not (s > r):
                            continue
                        else:
                            if idx1 == idx2 and idx1 not in used_tids:
                                used_tids.append(idx1)
                                # Test if invariant mass corresponds to pi0 mass (if not: likely to have matched wrong showers!)
                                Etot = (self.true_info['gamma_ekin'][r]+self.true_info['gamma_ekin'][s])**2
                                ptot = (self.true_info['gamma_mom'][r][0]+self.true_info['gamma_mom'][s][0])**2 +\
                                       (self.true_info['gamma_mom'][r][1]+self.true_info['gamma_mom'][s][1])**2 +\
                                       (self.true_info['gamma_mom'][r][2]+self.true_info['gamma_mom'][s][2])**2
                                invariant_mass = np.sqrt(Etot-ptot)
                                if invariant_mass < 0.95*134.9766 or invariant_mass > 1.05*134.9766:
                                    print(' WARNING: Pi0 mass deviates > 5% from literature value. Likely to have matched two photons of different pi0s!! ')
                                self.true_info['pi0_mass'].append(invariant_mass)
                                self.true_info['pi0_ekin'].append(self.true_info['gamma_ekin'][r]+self.true_info['gamma_ekin'][s]-invariant_mass)
                                #self.true_info['pi0_mom'].append() # TODO: Calculate it
                                costheta = np.dot(self.true_info['gamma_dir'][s],self.true_info['gamma_dir'][r])
                                if abs(costheta) <= 1.:
                                    self.true_info['gamma_angle'].append(np.arccos(costheta)) # append twice, once for every true photon
                                    self.true_info['gamma_angle'].append(np.arccos(costheta))
                                else:
                                    print(' WARNING: |costheta| > 1, cannot append true gamma_angle to the photons!! ')

        # Produce a list of n lists (n = n_gammas) with particle_IDs, particle_track_IDs and group_IDs of each shower
        if self.true_info['n_gammas'] > 0:
            self.true_info['shower_particle_ids']  = [[] for _ in range(self.true_info['n_gammas'])]
            #self.true_info['shower_particle_tids'] = [[] for _ in range(self.true_info['n_gammas'])]
            self.true_info['shower_group_ids']     = [[] for _ in range(self.true_info['n_gammas'])]
            counter = 0
            for i, p in enumerate(event['particles']):
                if p.parent_pdg_code() == 111 and p.pdg_code() == 22:             # p1 is a true photon
                    self.true_info['shower_particle_ids'][counter].append(p.id())
                    #self.true_info['shower_particle_tids'][counter].append(p.track_id())
                    counter += 1
            for i, p in enumerate(event['particles']):
                for gamma in range(self.true_info['n_gammas']):
                    if p.parent_id() in self.true_info['shower_particle_ids'][gamma] and p.id() not in self.true_info['shower_particle_ids'][gamma]:
                        self.true_info['shower_particle_ids'][gamma].append(p.id())
                    #if p.parent_track_id() in self.true_info['shower_particle_tids'][gamma] and p.track_id() not in self.true_info['shower_particle_tids'][gamma]:
                        #self.true_info['shower_particle_tids'][gamma].append(p.track_id())
            counter = 0
            for i, p in enumerate(event['particles']):
                if p.track_id() in tids_of_true_photons: # and p.pdg_code() == 22:
                    self.true_info['shower_group_ids'][counter].append(p.group_id())
                    counter += 1

        '''
        # Test if photons of true showers leave the detector -> missing energy
        for sh_index, particle_ids in enumerate(self.true_info['shower_particle_ids']):
            for index, p_ID in enumerate(particle_ids):
                for i, p in enumerate(event['particles']):
                    if p_ID == p.id() and p.pdg_code()==22:
                        if p.first_step().x()<0 or p.first_step().x()>767 or\
                           p.first_step().y()<0 or p.first_step().y()>767 or\
                           p.first_step().z()<0 or p.first_step().z()>767:
                            print(' photons first step outside of LAr volume: ')
                            print(' first_step: %.2f  %.2f  %.2f \t einit: %.2f \t mom: %.2f  %.2f  %.2f'\
                                  %(p.first_step().x(), p.first_step().y(), p.first_step().z(), p.energy_init(), p.px(), p.py(), p.pz()))
                        break
        '''

        # Test if photon makes compton scattering
        for i, p in enumerate(event['particles']):
            if p.parent_pdg_code()==22 and p.parent_id() in ids_of_true_photons:
                if p.parent_id() not in ids_of_photons_making_compton:
                    ids_of_photons_making_compton.append(p.parent_id())
                    compton_electron_first_step.append([p.first_step().x(),p.first_step().y(),p.first_step().z()]) #, p.first_step().t()])
        self.true_info['gamma_ids_making_compton_scat'] = ids_of_photons_making_compton
        self.true_info['compton_electron_first_step'] = compton_electron_first_step

        # Obtain shower's first (in time) energy deposit and define it as true shower's start position
        for j, showers_particle_ids in enumerate(self.true_info['shower_particle_ids']):
            min_time = float('inf')
            first_step = [float('inf'), float('inf'), float('inf')]
            for i, p in enumerate(event['particles']):
                if p.id() in showers_particle_ids:
                    if p.first_step().t() > 0. and p.first_step().t() < min_time:
                        min_time = p.first_step().t()
                        first_step = [p.first_step().x(), p.first_step().y(), p.first_step().z()] # , p.first_step().t()
            self.true_info['shower_first_edep'].append(first_step)

        # Loop over all clusters and get voxels and total deposited energy for every true gamma shower
        # Note: using parser 'parse_cluster3d_full', one can obtain a cluster via
        # clusters = event['cluster_label'] where the entries are
        # x,y,z,batch_id,voxel_value,cluster_id,group_id,semantic_type
        if self.true_info['n_gammas'] > 0:
            self.true_info['gamma_voxels'] = [[] for _ in range(self.true_info['n_gammas'])]
            self.true_info['gamma_edep']   = [0. for _ in range(self.true_info['n_gammas'])]
            # gamma_voxels is a list of n lists (n = number of gammas) with voxel coordinates of each shower
            clusters = event['cluster_label']
            for cluster_index, edep in enumerate(clusters):
                for group_index, group in enumerate(self.true_info['shower_group_ids']):
                    summed_edep = 0.
                    if edep[6] == group[0]:
                        self.true_info['gamma_voxels'][group_index].append([edep[0],edep[1],edep[2]])
                        self.true_info['gamma_edep'][group_index] += edep[4]
            for index, gamma in enumerate(self.true_info['gamma_voxels']):
                self.true_info['gamma_n_voxels'].append(len(self.true_info['gamma_voxels'][index]))
        else:
            self.true_info['gamma_voxels'] = []

        # Out-Of-Fiducial-Volume (OOFV) information:
        # If at least one edep is OOFV: Put shower number to list output['OOFV']
        # This is relatively strict -> might want to add the shower to OOFV
        # only if a certain fraction of all edeps is OOFV
        # TODO: Also add the shower number to OOFV if true_gamma.first_step is OOFV (otherwise it could happen that a true photon which leaves the LAr volume without edep is not OOFV)
        for shower_index, shower in enumerate(self.true_info['gamma_voxels']):
            for edep in range(len(shower)):
                coordinate = np.array((shower[edep][0],shower[edep][1],shower[edep][2]))
                if ( np.any(coordinate<fiducial) or np.any(coordinate>(767-fiducial)) ):
                    self.true_info['OOFV'].append(shower_index)
                    break
        return


    def extract_reco_information(self, event, output):
        '''
        Obtain reconstructed informations about pi0s and gammas originated from pi0 decays and dump
        it to self.reco_info['<variable>']
        '''

        self.reco_info['ev_id']    = event['index']
        self.reco_info['n_pi0']    = len(output['matches'])
        self.reco_info['n_gammas'] = 2.*len(output['matches'])
        self.reco_info['OOFV']     = output['OOFV']

        print(' ============ n reco pi0: ', self.reco_info['n_pi0'])

        showers = output['showers']

        # Note: match = if two showers point to the same point and this point is close to a track
        for match in range(self.reco_info['n_pi0']):
            match_1 = output['matches'][match][0]
            match_2 = output['matches'][match][1]
            self.reco_info['matches'].append(match_1)
            self.reco_info['matches'].append(match_2)
            self.reco_info['gamma_mom'].append(np.array(showers[match_1].direction*showers[match_1].energy))
            self.reco_info['gamma_mom'].append(np.array(showers[match_2].direction*showers[match_2].energy))
            self.reco_info['gamma_dir'].append(np.array(showers[match_1].direction))
            self.reco_info['gamma_dir'].append(np.array(showers[match_2].direction))
            self.reco_info['gamma_start'].append(np.array(showers[match_1].start))
            self.reco_info['gamma_start'].append(np.array(showers[match_2].start))
            self.reco_info['gamma_edep'].append(showers[match_1].energy)
            self.reco_info['gamma_edep'].append(showers[match_2].energy)
            self.reco_info['gamma_pid'].append(showers[match_1].pid)
            self.reco_info['gamma_pid'].append(showers[match_2].pid)
            self.reco_info['gamma_voxels_mask'].append(np.array(showers[match_1].voxels))
            self.reco_info['gamma_voxels_mask'].append(np.array(showers[match_2].voxels))
            self.reco_info['gamma_n_voxels_mask'].append(showers[match_1].voxels.size)
            self.reco_info['gamma_n_voxels_mask'].append(showers[match_2].voxels.size)

            # Obtain the showers edeps (x,y,z,batch_id,energy_deposition)
            mask = output['shower_mask']                           # mask for all edeps classified as shower
            voxels_1 = output['showers'][match_1].voxels           # indices in the mask for the 1st match
            voxels_2 = output['showers'][match_2].voxels           # indices in the mask for the 2nd match
            edeps_1 = output['energy'][mask][voxels_1]             # all edeps for the 1st match
            edeps_2 = output['energy'][mask][voxels_2]             # all edeps for the 2nd match
            self.reco_info['gamma_voxels'].append(np.array(edeps_1))
            self.reco_info['gamma_voxels'].append(np.array(edeps_2))
            self.reco_info['gamma_n_voxels'].append(len(edeps_1))
            self.reco_info['gamma_n_voxels'].append(len(edeps_2))

        # Reconstructed angle and pi0 mass
        for match in output['matches']:
            idx1, idx2 = match
            s1, s2 = output['showers'][idx1], output['showers'][idx2]
            e1, e2 = s1.energy, s2.energy
            t1, t2 = s1.direction, s2.direction
            #if np.any(np.isnan(t1)) or np.any(np.isnan(t2)):
            #    print(' WARNING: shower direction not assigned: \t dir_1: ', t1, ' \t dir_2: ', t2)
            #    print(' \t -> set costheta = 1 and pi0_mass = 0 ')
            #    costheta = 1.
            #else:
            try:
                costheta = np.dot(t1, t2)
            except:
                print(' WARNING: shower direction not assigned: \t dir_1: ', t1, ' \t dir_2: ', t2)
                print(' \t -> set costheta = 1 and pi0_mass = 0 ')
                costheta = 1.

            if abs(costheta) > 1.:
                print(' WARNING: costheta = np.dot(sh1.dir, sh2.dir) = ', costheta, ' > 1.')
                print(' \t -> set costheta = 1 and pi0_mass = 0 ')
                costheta = 1
            self.reco_info['gamma_angle'].append(np.arccos(costheta))
            self.reco_info['gamma_angle'].append(np.arccos(costheta))
            self.reco_info['pi0_mass'].append(math.sqrt(2.*e1*e2*(1.-costheta)))
            self.reco_info['pi0_mass'].append(math.sqrt(2.*e1*e2*(1.-costheta)))
        return


    def find_true_electron_showers(self, event, output):
        '''
        Obtain true information about showers induced by (primary) electrons and dump it to output['electronShowers'].
        '''
        #print(' ---- in function find_true_electron_showers ---- ')
        electron_showers = []

        # Get group_id of shower produced by electron
        group_ids = []
        #ids = []
        #tids = []
        for i, p in enumerate(event['particles']):
            if (p.pdg_code() == 11 or p.pdg_code() == -11) and p.parent_pdg_code() == 0:
                #print(p.dump())
                #print(' --- p.track_id: ', p.track_id())
                #print(' --- p.id: ', p.id())
                #print(' --- pos: ', p.x(), p.y(), p.z(), p.t())
                #print(' --- first_step: ', p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t())
                if p.group_id() not in group_ids:
                    group_ids.append(p.group_id())
                    #ids.append([p.id()])
                    #tids.append([p.track_id()])
        #print(' group_ids: ', group_ids)
        #print(' ids: ', ids)
        #print(' tids: ', tids)

        '''
        for list_index, id_list in enumerate(ids):
            for i, p in enumerate(event['particles']):
                if p.parent_id() in id_list:
                    id_list.append(p.id())
        print(' ids: ', ids)

        for list_index, tid_list in enumerate(tids):
            for i, p in enumerate(event['particles']):
                if p.parent_track_id() in tid_list:
                    tid_list.append(p.track_id())
        print(' tids: ', tids)
        '''

        # Find earliest (in time) edep
        earliest_edep_pos = []
        for index, group_id in enumerate(group_ids):
            min_time = float('inf')
            pos = [float('inf'),float('inf'),float('inf')]
            #earliest_edep_pos = [float('inf'), float('inf'), float('inf')]
            for i, p in enumerate(event['particles']):
                if p.group_id() == group_id:
                    # consider time and take earliest
                    #print(' p.first_step: \t x: %.2f \t y: %.2f \t z: %.2f \t t: %.2f' %(p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()))
                    if p.first_step().t() >= 0. and p.first_step().t() < min_time:
                        min_time = p.first_step().t()
                        pos = [p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()]
            if np.any(np.isinf(pos)):
                print(' Shower has no physical edep [event:', event['index'], ']')
            earliest_edep_pos.append(pos)
            #print(' earliest: ', pos)

        # Loop over all clusters and get voxels for every electron induced shower
        # Note: using parser 'parse_cluster3d_full', one can obtain a cluster via
        # clusters = event['cluster_label'] where the entries are
        # x,y,z,batch_id,voxel_value,cluster_id,group_id,semantic_type
        voxels = [[] for _ in range(len(group_ids))]
        edeps  = [[] for _ in range(len(group_ids))]
        clusters = event['cluster_label']
        for cluster_index, edep in enumerate(clusters):
            for group_index, group in enumerate(group_ids):
                if edep[6] == group:
                    voxels[group_index].append([edep[0],edep[1],edep[2]])
                    edeps[group_index].append(edep[4])

        # Assign the electron induced shower's properties
        counter = 0
        for i, p in enumerate(event['particles']):
            #print(p.dump())
            if (p.pdg_code() == 11 or p.pdg_code() == -11) and p.parent_pdg_code() == 0:
                _pid        = int(p.id())
                _group_id   = p.group_id()
                _start      = [p.x(), p.y(), p.z(), p.t()] # creation vertex position (x,y,z,t)
                _first_step = [p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()] # first step (x,y,z,t)
                _first_edep = earliest_edep_pos[group_ids.index(p.group_id())] # earliest (in time) edep position (x,y,z,t)
                _mom_init   = [p.px(), p.py(), p.pz()]
                _dir        = [p.px(), p.py(), p.pz()]/np.linalg.norm([p.px(), p.py(), p.pz()])
                _voxels     = voxels[counter]
                _einit      = p.energy_init()
                _edeps      = edeps[counter]
                _edep_tot   = np.sum(edeps[counter])
                # TODO: check that shower starts within FV, add attribute if needed
                # TODO: check that only edeps within LAr volume are counted
                electron_showers.append(ElectronShower(pid=_pid,group_id=_group_id,start=_start,first_step=_first_step,first_edep=_first_edep,mom_init=_mom_init,direction=_dir,voxels=_voxels,einit=_einit,edeps=_edeps,edep_tot=_edep_tot))
                counter += 1
        output['electronShowers'] = electron_showers
        return


    def find_true_photon_showers(self, event, output):
        '''
        Obtain true information about showers induced by photons from a pi0 decay and dump it to output['photonShowers'].
        '''
        #print(' ========================================================================== ')
        #print(' event_id: ', event['index'])
        photon_showers  = []
        compton_showers = []

        # First, produce list of n lists (n=number of photons from pi0 decay) with ids, track_ids and group_ids
        ids_of_true_photons  = []
        tids_of_true_photons = []
        group_ids            = []
        for i, p in enumerate(event['particles']):
            if p.parent_pdg_code() == 111 and p.pdg_code() == 22:
                ids_of_true_photons.append(p.id())
                tids_of_true_photons.append(p.track_id())
                group_ids.append(p.group_id())
        #print(' ids_of_true_photons:           ', ids_of_true_photons)
        #print(' tids_of_true_photons:          ', tids_of_true_photons)
        #print(' group_ids:                     ', group_ids)

        # Loop over all clusters and get voxels for every photon induced shower
        # Note: using parser 'parse_cluster3d_full', one can obtain a cluster via
        # clusters = event['cluster_label'] where the entries are
        # x,y,z,batch_id,voxel_value,cluster_id,group_id,semantic_type
        voxels = [[] for _ in range(len(group_ids))]
        edeps  = [[] for _ in range(len(group_ids))]
        clusters = event['cluster_label']
        for cluster_index, edep in enumerate(clusters):
            for group_index, group in enumerate(group_ids):
                if edep[6] == group:
                    voxels[group_index].append([edep[0],edep[1],edep[2]])
                    edeps[group_index].append(edep[4])

        # Produce list of n lists (n=number of photons from pi0 decay) with all ids and track_ids of particles belonging to the same shower
        ids_of_particles_in_shower  = [[ID] for counter, ID in enumerate(ids_of_true_photons)]
        tids_of_particles_in_shower = [[ID] for counter, ID in enumerate(ids_of_true_photons)]
        for sh_index in range(len(ids_of_true_photons)):
            for i, p in enumerate(event['particles']):
                if p.parent_id() in ids_of_particles_in_shower[sh_index] and p.id() not in ids_of_particles_in_shower[sh_index]:
                    ids_of_particles_in_shower[sh_index].append(p.id())
                if p.parent_track_id() in tids_of_particles_in_shower[sh_index] and p.track_id() not in tids_of_particles_in_shower[sh_index]:
                    tids_of_particles_in_shower[sh_index].append(p.track_id())
        #print(' ids_of_particles_in_shower:    ', ids_of_particles_in_shower)
        #print(' tids_of_particles_in_shower:   ', tids_of_particles_in_shower)

        # Get IDs and track IDs of photons which make compton scattering (note: including photons produced in EM shower, not only photons from pi0 decays)
        ids_of_photons_making_compton  = []
        tids_of_photons_making_compton = []
        for i, p in enumerate(event['particles']):
            if p.parent_id() in ids_of_true_photons:
                #print(' PARENT (', p.parent_id(), ') in ids_of_true_photons... ')
                if p.pdg_code() == 11:
                    if p.parent_id() not in ids_of_photons_making_compton:
                        ids_of_photons_making_compton.append(p.parent_id())
                    if p.parent_track_id() not in tids_of_photons_making_compton:
                        tids_of_photons_making_compton.append(p.parent_track_id())
                    #print(' Photon makes compton scattering ')
                    #print(p.dump())
                    #print(' p.id:              ', p.id())
                    #print(' p.track_id:        ', p.track_id())
                    #print(' p.group_id:        ', p.group_id())
                    #print(' p.pdg_code:        ', p.pdg_code())
                    #print(' p.parent_id:       ', p.parent_id())
                    #print(' p.parent_track_id: ', p.parent_track_id())
                    #print(' p.parent_pdg_code: ', p.parent_pdg_code())
                    #print(' p.first_step: \t x: %.2f \t y: %.2f \t z: %.2f \t t: %.2f' %(p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()))
                    #compton_electron_first_step.append([p.first_step().x(),p.first_step().y(),p.first_step().z()]) #, p.first_step().t()])
        #print(' ids_of_photons_making_compton: ', ids_of_photons_making_compton)
        #print(' tids_of_photons_making_compton: ', tids_of_photons_making_compton)

        # Get IDs and track IDs of photons which make compton scattering (note: including those compton scattered electrons produced in an EM shower, not only those scattered with a photon from a pi0 decay)
        compton_electron_tids = []
        for i, TID in enumerate(tids_of_photons_making_compton):
            for j, p in enumerate(event['particles']):
                if p.parent_track_id() == TID and p.track_id() not in compton_electron_tids:
                    compton_electron_tids.append(p.track_id())
        #print(' compton_electron_tids: ', compton_electron_tids)

        # Get track IDs of photons which of photons which make compton scattering (note: in contrast to above, only photons from pi0 decays which make compton scattering are included here)
        tids_of_photons_with_primary_compton = []
        for i, TID in enumerate(compton_electron_tids):
            for j, p in enumerate(event['particles']):
                #if p.track_id() == TID:
                    #print(p.dump())
                if p.parent_track_id() in tids_of_true_photons:
                    if p.parent_track_id() not in tids_of_photons_with_primary_compton:
                        tids_of_photons_with_primary_compton.append(p.parent_track_id())
        #if len(tids_of_photons_with_primary_compton) > 0:
        #    print(' tids_of_photons_with_primary_compton: ', tids_of_photons_with_primary_compton)

        # Obtain shower's first (in time) energy deposit and define it as true shower's start position
        earliest_edep_pos = []
        for j, IDs in enumerate(ids_of_particles_in_shower):
            min_time = float('inf')
            first_step = [float('inf'), float('inf'), float('inf')]
            for i, p in enumerate(event['particles']):
                if p.id() in IDs:
                    #print(' p.first_step: \t x: %.2f \t y: %.2f \t z: %.2f \t t: %.2f' %(p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()))
                    first_step = [p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()]
                    if p.first_step().t() > 0.:
                        if p.first_step().t() < min_time:
                            min_time = p.first_step().t()
                            first_step = [p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()]
            earliest_edep_pos.append(first_step)

        # Assign the photon induced shower's properties
        counter = 0
        for i, ID in enumerate(ids_of_true_photons):
            for j, p in enumerate(event['particles']):
                if p.id() == ID:
                    _pid        = int(p.id())
                    _group_id   = p.group_id()
                    _start      = [p.x(), p.y(), p.z(), p.t()] # creation vertex position (x,y,z,t)
                    _first_step = [p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()] # first step (x,y,z,t)
                    _first_edep = earliest_edep_pos[group_ids.index(p.group_id())] # earliest (in time) edep position (x,y,z,t)
                    _mom_init   = [p.px(), p.py(), p.pz()]
                    _dir        = [p.px(), p.py(), p.pz()]/np.linalg.norm([p.px(), p.py(), p.pz()])
                    _voxels     = voxels[counter]
                    _einit      = p.energy_init()
                    _edeps      = edeps[counter]
                    _edep_tot   = np.sum(edeps[counter])
                    # TODO: check that shower starts within FV, add attribute if needed
                    # TODO: check that only edeps within LAr volume are counted
                    photon_showers.append(PhotonShower(pid=_pid,group_id=_group_id,start=_start,first_step=_first_step,first_edep=_first_edep,mom_init=_mom_init,direction=_dir,voxels=_voxels,einit=_einit,edeps=_edeps,edep_tot=_edep_tot))
                    counter += 1
                    break
        output['photonShowers'] = photon_showers

        # Assign the photon induced shower's properties (for showers where the initiating photon makes Compton Scattering)
        counter = 0
        for i, TID in enumerate(tids_of_photons_with_primary_compton):
            for j, p in enumerate(event['particles']):
                if p.track_id() == TID:
                    _pid        = int(p.id())
                    _group_id   = p.group_id()
                    _start      = [p.x(), p.y(), p.z(), p.t()] # creation vertex position (x,y,z,t)
                    _first_step = [p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()] # first step (x,y,z,t)
                    _first_edep = earliest_edep_pos[group_ids.index(p.group_id())] # earliest (in time) edep position (x,y,z,t)
                    _mom_init   = [p.px(), p.py(), p.pz()]
                    _dir        = [p.px(), p.py(), p.pz()]/np.linalg.norm([p.px(), p.py(), p.pz()])
                    _voxels     = voxels[counter]
                    _einit      = p.energy_init()
                    _edeps      = edeps[counter]
                    _edep_tot   = np.sum(edeps[counter])
                    # TODO: check that shower starts within FV, add attribute if needed
                    # TODO: check that only edeps within LAr volume are counted
                    compton_showers.append(ComptonShower(pid=_pid,group_id=_group_id,start=_start,first_step=_first_step,first_edep=_first_edep,mom_init=_mom_init,direction=_dir,voxels=_voxels,einit=_einit,edeps=_edeps,edep_tot=_edep_tot))
                    counter += 1
                    break
        output['comptonShowers'] = compton_showers

        return


    def extract_reco_information(self, event, output):
        '''
        Obtain reconstructed informations about pi0s and gammas originated from pi0 decays and dump
        it to self.reco_info['<variable>']
        '''

        self.reco_info['ev_id']                  = event['index']              # [-]
        self.reco_info['n_pi0']                  = len(output['matches'])      # [-]
        self.reco_info['n_gammas']               = 2.*len(output['matches'])   # [-]
        #self.reco_info['matches']                = []                               # [-]
        #self.reco_info['gamma_mom']              = []                               # [MeV/c]
        #self.reco_info['gamma_dir']              = []                               # [x,y,z]
        #self.reco_info['gamma_start']            = []                               # [x,y,z] # pi0->2gamma vertex
        #self.reco_info['gamma_edep']             = []                               # [MeV]
        #self.reco_info['gamma_pid']              = []                               # [-]
        #self.reco_info['gamma_voxels_mask']      = []                               # [-]
        #self.reco_info['gamma_n_voxels_mask']    = []                               # [-]
        #self.reco_info['gamma_voxels']           = []                               # [-]
        #self.reco_info['gamma_n_voxels']         = []                               # [-]
        #self.reco_info['gamma_angle']            = []                               # [rad]
        #self.reco_info['pi0_mass']               = []                               # [MeV/c2]
        self.reco_info['OOFV']                   = output['OOFV']              # [-]

        showers = output['showers']

        # Note: match = if two showers point to the same point and this point is close to a track
        for match in range(self.reco_info['n_pi0']):
            match_1 = output['matches'][match][0]
            match_2 = output['matches'][match][1]
            self.reco_info['matches'].append(match_1)
            self.reco_info['matches'].append(match_2)
            self.reco_info['gamma_mom'].append(np.array(showers[match_1].direction*showers[match_1].energy))
            self.reco_info['gamma_mom'].append(np.array(showers[match_2].direction*showers[match_2].energy))
            self.reco_info['gamma_dir'].append(np.array(showers[match_1].direction))
            self.reco_info['gamma_dir'].append(np.array(showers[match_2].direction))
            self.reco_info['gamma_start'].append(np.array(showers[match_1].start))
            self.reco_info['gamma_start'].append(np.array(showers[match_2].start))
            self.reco_info['gamma_edep'].append(showers[match_1].energy)
            self.reco_info['gamma_edep'].append(showers[match_2].energy)
            self.reco_info['gamma_pid'].append(showers[match_1].pid)
            self.reco_info['gamma_pid'].append(showers[match_2].pid)
            self.reco_info['gamma_voxels_mask'].append(np.array(showers[match_1].voxels))
            self.reco_info['gamma_voxels_mask'].append(np.array(showers[match_2].voxels))
            self.reco_info['gamma_n_voxels_mask'].append(showers[match_1].voxels.size)
            self.reco_info['gamma_n_voxels_mask'].append(showers[match_2].voxels.size)

            # Obtain the showers edeps (x,y,z,batch_id,energy_deposition)
            voxels_1 = output['showers'][match_1].voxels           # indices in the mask for the 1st match
            voxels_2 = output['showers'][match_2].voxels           # indices in the mask for the 2nd match
            edeps_1 = output['energy'][voxels_1]             # all edeps for the 1st match
            edeps_2 = output['energy'][voxels_2]             # all edeps for the 2nd match
            self.reco_info['gamma_voxels'].append(np.array(edeps_1))
            self.reco_info['gamma_voxels'].append(np.array(edeps_2))
            self.reco_info['gamma_n_voxels'].append(len(edeps_1))
            self.reco_info['gamma_n_voxels'].append(len(edeps_2))

        # Reconstructed angle and pi0 mass
        for match in output['matches']:
            idx1, idx2 = match
            s1, s2 = output['showers'][idx1], output['showers'][idx2]
            e1, e2 = s1.energy, s2.energy
            t1, t2 = s1.direction, s2.direction
            #if np.any(np.isnan(t1)) or np.any(np.isnan(t2)):
            #    print(' WARNING: shower direction not assigned: \t dir_1: ', t1, ' \t dir_2: ', t2)
            #    print(' \t -> set costheta = 1 and pi0_mass = 0 ')
            #    costheta = 1.
            #else:
            try:
                costheta = np.dot(t1, t2)
            except:
                print(' WARNING: shower direction not assigned: \t dir_1: ', t1, ' \t dir_2: ', t2)
                print(' \t -> set costheta = 1 and pi0_mass = 0 ')
                costheta = 1.

            if abs(costheta) > 1.:
                print(' WARNING: costheta = np.dot(sh1.dir, sh2.dir) = ', costheta, ' > 1.')
                print(' \t -> set costheta = 1 and pi0_mass = 0 ')
                costheta = 1
            self.reco_info['gamma_angle'].append(np.arccos(costheta))
            self.reco_info['gamma_angle'].append(np.arccos(costheta))
            self.reco_info['pi0_mass'].append(math.sqrt(2.*e1*e2*(1.-costheta)))
            self.reco_info['pi0_mass'].append(math.sqrt(2.*e1*e2*(1.-costheta)))
        return

    
# Classes that contain the PPN predicted points (one class for every semantic type)
# TODO: Instead of having 5 classes for every semantic type, make one class 'PPN_Predictions' with all shower, track, michel, delta, LEScatter predictions
class ShowerPoints():
    def __init__(self, ppns=-np.ones(3), shower_score=-1., shower_id=-1):
        self.ppns = ppns
        self.shower_score = shower_score
        self.shower_id = int(shower_id)
    def __str__(self):
        return """ Track  ID {}
        PPN point  : ({:0.2f},{:0.2f},{:0.2f})
        Shower score: {}""".format(self.shower_id, *self.ppns, self.shower_score)
class TrackPoints():
    def __init__(self, ppns=-np.ones(3), track_score=-1., track_id=-1):
        self.ppns = ppns
        self.track_score = track_score
        self.track_id = int(track_id)
    def __str__(self):
        return """ Track  ID {}
        PPN point  : ({:0.2f},{:0.2f},{:0.2f})
        Track score: {}""".format(self.track_id, *self.ppns, self.track_score)
class MichelPoints():
    def __init__(self, ppns=-np.ones(3), michel_score=-1., michel_id=-1):
        self.ppns = ppns
        self.michel_score = michel_score
        self.michel_id = int(michel_id)
    def __str__(self):
        return """ Track  ID {}
        PPN point  : ({:0.2f},{:0.2f},{:0.2f})
        Michel score: {}""".format(self.michel_id, *self.ppns, self.michel_score)
class DeltaPoints():
    def __init__(self, ppns=-np.ones(3), delta_score=-1., delta_id=-1):
        self.ppns = ppns
        self.delta_score = delta_score
        self.delta_id = int(delta_id)
    def __str__(self):
        return """ Track  ID {}
        PPN point  : ({:0.2f},{:0.2f},{:0.2f})
        Delta score: {}""".format(self.delta_id, *self.ppns, self.delta_score)
class LEScatPoints():
    def __init__(self, ppns=-np.ones(3), LEScat_score=-1., LEScat_id=-1):
        self.ppns = ppns
        self.LEScat_score = LEScat_score
        self.LEScat_id = int(LEScat_id)
    def __str__(self):
        return """ Track  ID {}
        PPN point  : ({:0.2f},{:0.2f},{:0.2f})
        LEScat score: {}""".format(self.LEScat_id, *self.ppns, self.LEScat_score)
