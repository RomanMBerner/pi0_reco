import math
import numpy as np


class ElectronShower(): # Including positron induced showers
    def __init__(self, pid=-1, group_id=-1, start=-np.ones(4), first_step=-np.ones(4), first_edep=-np.ones(4), mom_init=-np.ones(3), direction=-np.ones(3), voxels=[], einit=-1, edeps=[], edep_tot=-1):
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
    def __init__(self, pid=-1, group_id=-1, start=-np.ones(4), first_step=-np.ones(4), first_edep=-np.ones(4), mom_init=-np.ones(3), direction=-np.ones(3), voxels=[], einit=-1, edeps=[], edep_tot=-1):
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
    def __init__(self, pid=-1, group_id=-1, start=-np.ones(4), first_step=-np.ones(4), first_edep=-np.ones(4), mom_init=-np.ones(3), direction=-np.ones(3), voxels=[], einit=-1, edeps=[], edep_tot=-1):
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
    
    def extract_true_information(self, event):
        '''
        Obtain true informations about pi0s and gammas originated from pi0 decays and dump
        it to self.true_info['<variable>']
        '''
        
        # Defined objects
        self.true_info['ev_id']                         = self.event['index'] # [-]
        self.true_info['n_pi0']                         = 0                   # [-]
        self.true_info['n_gammas']                      = 0                   # [-]
        self.true_info['pi0_track_ids']                 = []                  # [-]
        self.true_info['gamma_group_ids']               = []                  # [-]
        self.true_info['shower_particle_ids']           = []                  # [-]
        #self.true_info['gamma_ids_making_compton_scat']                      # [-] # List of photon ids which make compton scattering
        self.true_info['pi0_ekin']                      = []                  # [MeV]
        self.true_info['pi0_mass']                      = []                  # [MeV/c2] # Calculated with energy of the gammas and their momenta (invariant mass = sqrt(Etot-ptot))
        self.true_info['gamma_pos']                     = []                  # [x,y,z] # pi0 -> gamma+gamma vertex
        self.true_info['gamma_dir']                     = []                  # [x,y,z]
        self.true_info['gamma_mom']                     = []                  # [MeV/c]
        self.true_info['gamma_ekin']                    = []                  # [MeV] # initial energy of photon, = np.sqrt(p.px()**2+p.py()**2+p.pz()**2)
        self.true_info['gamma_edep']                    = []                  # [MeV]
        #self.true_info['gamma_voxels']                                       # List lists (for every shower 1 list) of voxels containing edeps
        self.true_info['gamma_n_voxels']                = []                  # [-] # Number of voxels containing edeps for each shower
        self.true_info['gamma_first_step']              = []                  # [x,y,z] # Pos of the photon's 1st energy deposition
        #self.true_info['compton_electron_first_step']                        # [x,y,z] # Pos of the compton electron's 1st energy deposition
        self.true_info['shower_first_edep']             = []                  # [x,y,z] # Pos of a shower's 1st (in time) energy deposition
        self.true_info['OOFV']                          = []                  # [-] # If shower has edep(s) close to the LAr volume boundary.
                                                                              # Note: If photon leaves detector without producing an edep, this is NOT classified as OOFV.
                                                                              # For those events: Consider self.true_info['n_voxels']
        self.true_info['gamma_angle']                   = []                  # [rad] # Opening angle between the two photons from a pi0 decay

        # Define some lists for pi0s, daughter photons and their compton electrons
        ids_of_true_photons           = [] # every photon has its own id
        tids_of_true_photons          = [] # every photon has its own tid
        parent_tids_of_true_photons   = [] # photons from the same pi0 decay have the same parent_track_id
        ids_of_photons_making_compton = [] # True or False for every true photon
        compton_electron_first_step   = [] # [x,y,z] coordinates of compton electrons first step
        
        # Get photons from pi0 decays:
        # Note: Photons from the same pi0 have the same parent_track_id.
        #       Every photon (from pi0 decay) has its own id.
        #       Primary pi0s do not have id and track_id
        for i, p in enumerate(self.event['particles'][0]):
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
            for i, p in enumerate(self.event['particles'][0]):
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
        
        # Produce a list of n lists (n = n_gammas) with particle_IDs and group_IDs of each shower
        if self.true_info['n_gammas'] > 0:
            self.true_info['shower_particle_ids'] = [[] for _ in range(self.true_info['n_gammas'])]
            self.true_info['shower_group_ids']    = [[] for _ in range(self.true_info['n_gammas'])]
            counter = 0
            for i, p in enumerate(self.event['particles'][0]):
                if p.parent_pdg_code() == 111 and p.pdg_code() == 22:             # p1 is a true photon
                    self.true_info['shower_particle_ids'][counter].append(p.id())
                    counter += 1
            for i, p in enumerate(self.event['particles'][0]):
                for gamma in range(self.true_info['n_gammas']):
                    if p.parent_id() in self.true_info['shower_particle_ids'][gamma] and p.id() not in self.true_info['shower_particle_ids'][gamma]:
                        self.true_info['shower_particle_ids'][gamma].append(p.id())
            counter = 0
            for i, p in enumerate(self.event['particles'][0]):
                if p.track_id() in tids_of_true_photons: # and p.pdg_code() == 22:
                    self.true_info['shower_group_ids'][counter].append(p.group_id())
                    counter += 1
        
        # Test if photon makes compton scattering
        for i, p in enumerate(self.event['particles'][0]):
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
            for i, p in enumerate(self.event['particles'][0]):
                if p.id() in showers_particle_ids:
                    if p.first_step().t() > 0. and p.first_step().t() < min_time:
                        min_time = p.first_step().t()
                        first_step = [p.first_step().x(), p.first_step().y(), p.first_step().z()] # , p.first_step().t()
            self.true_info['shower_first_edep'].append(first_step)
        
        # Loop over all clusters and get voxels and total deposited energy for every true gamma shower
        # Note: using parser 'parse_cluster3d_full', one can obtain a cluster via
        # clusters = self.event['cluster_label'] where the entries are
        # x,y,z,batch_id,voxel_value,cluster_id,group_id,semantic_type
        if self.true_info['n_gammas'] > 0:
            self.true_info['gamma_voxels'] = [[] for _ in range(self.true_info['n_gammas'])]
            self.true_info['gamma_edep']   = [0. for _ in range(self.true_info['n_gammas'])]
            # gamma_voxels is a list of n lists (n = number of gammas) with voxel coordinates of each shower
            clusters = self.event['cluster_label']
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
        # If at least one edep is OOFV: Put shower number to list self.output['OOFV']
        # This is relatively strict -> might want to add the shower to OOFV
        # only if a certain fraction of all edeps is OOFV
        # TODO: Also add the shower number to OOFV if true_gamma.first_step is OOFV (otherwise it could happen that a true photon which leaves the LAr volume without edep is not OOFV)
        for shower_index, shower in enumerate(self.true_info['gamma_voxels']):
            for edep in range(len(shower)):
                coordinate = np.array((shower[edep][0],shower[edep][1],shower[edep][2]))
                if ( np.any(coordinate<self.cfg['fiducialize']) or np.any(coordinate>(767-self.cfg['fiducialize'])) ):
                    self.true_info['OOFV'].append(shower_index)
                    break
        return
    
    
    def find_true_electron_showers(self, event):
        '''
        Obtain true information about showers induced by (primary) electrons and dump it to self.output['electronShowers'].
        '''        
        #print(' ---- in function find_true_electron_showers ---- ')
        electron_showers = []
        
        # Get group_id of shower produced by electron
        group_ids = []
        #ids = []
        #tids = []
        for i, p in enumerate(self.event['particles'][0]):
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
            for i, p in enumerate(self.event['particles'][0]):
                if p.parent_id() in id_list:
                    id_list.append(p.id())
        print(' ids: ', ids)
        
        for list_index, tid_list in enumerate(tids):
            for i, p in enumerate(self.event['particles'][0]):
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
            for i, p in enumerate(self.event['particles'][0]):
                if p.group_id() == group_id:
                    # consider time and take earliest
                    #print(' p.first_step: \t x: %.2f \t y: %.2f \t z: %.2f \t t: %.2f' %(p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()))
                    if p.first_step().t() >= 0. and p.first_step().t() < min_time:
                        min_time = p.first_step().t()
                        pos = [p.first_step().x(), p.first_step().y(), p.first_step().z(), p.first_step().t()]
            if np.any(np.isinf(pos)):
                print(' Shower has no physical edep [event:', self.event['index'], ']')
            earliest_edep_pos.append(pos)
            #print(' earliest: ', pos)
        
        # Loop over all clusters and get voxels for every electron induced shower
        # Note: using parser 'parse_cluster3d_full', one can obtain a cluster via
        # clusters = self.event['cluster_label'] where the entries are
        # x,y,z,batch_id,voxel_value,cluster_id,group_id,semantic_type
        voxels = [[] for _ in range(len(group_ids))]
        edeps  = [[] for _ in range(len(group_ids))]
        clusters = self.event['cluster_label']
        for cluster_index, edep in enumerate(clusters):
            for group_index, group in enumerate(group_ids):
                if edep[6] == group:
                    voxels[group_index].append([edep[0],edep[1],edep[2]])
                    edeps[group_index].append(edep[4])
        
        # Assign the electron induced shower's properties
        counter = 0
        for i, p in enumerate(self.event['particles'][0]):
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
        self.output['electronShowers'] = electron_showers
        return
        
        
    def find_true_photon_showers(self, event):
        '''
        Obtain true information about showers induced by photons from a pi0 decay and dump it to self.output['photonShowers'].
        '''
        #print(' ========================================================================== ')
        #print(' event_id: ', self.event['index'])
        photon_showers  = []
        compton_showers = []
        
        # First, produce list of n lists (n=number of photons from pi0 decay) with ids, track_ids and group_ids
        ids_of_true_photons  = []
        tids_of_true_photons = []
        group_ids            = []
        for i, p in enumerate(self.event['particles'][0]):
            if p.parent_pdg_code() == 111 and p.pdg_code() == 22:
                ids_of_true_photons.append(p.id())
                tids_of_true_photons.append(p.track_id())
                group_ids.append(p.group_id())
        #print(' ids_of_true_photons:           ', ids_of_true_photons)
        #print(' tids_of_true_photons:          ', tids_of_true_photons)
        #print(' group_ids:                     ', group_ids)
        
        # Loop over all clusters and get voxels for every photon induced shower
        # Note: using parser 'parse_cluster3d_full', one can obtain a cluster via
        # clusters = self.event['cluster_label'] where the entries are
        # x,y,z,batch_id,voxel_value,cluster_id,group_id,semantic_type
        voxels = [[] for _ in range(len(group_ids))]
        edeps  = [[] for _ in range(len(group_ids))]
        clusters = self.event['cluster_label']
        for cluster_index, edep in enumerate(clusters):
            for group_index, group in enumerate(group_ids):
                if edep[6] == group:
                    voxels[group_index].append([edep[0],edep[1],edep[2]])
                    edeps[group_index].append(edep[4])

        # Produce list of n lists (n=number of photons from pi0 decay) with all ids and track_ids of particles belonging to the same shower
        ids_of_particles_in_shower  = [[ID] for counter, ID in enumerate(ids_of_true_photons)]
        tids_of_particles_in_shower = [[ID] for counter, ID in enumerate(ids_of_true_photons)]
        for sh_index in range(len(ids_of_true_photons)):
            for i, p in enumerate(self.event['particles'][0]):
                if p.parent_id() in ids_of_particles_in_shower[sh_index] and p.id() not in ids_of_particles_in_shower[sh_index]:
                    ids_of_particles_in_shower[sh_index].append(p.id())
                if p.parent_track_id() in tids_of_particles_in_shower[sh_index] and p.track_id() not in tids_of_particles_in_shower[sh_index]:
                    tids_of_particles_in_shower[sh_index].append(p.track_id())
        #print(' ids_of_particles_in_shower:    ', ids_of_particles_in_shower)
        #print(' tids_of_particles_in_shower:   ', tids_of_particles_in_shower)
        
        # Get IDs and track IDs of photons which make compton scattering (note: including photons produced in EM shower, not only photons from pi0 decays)
        ids_of_photons_making_compton  = []
        tids_of_photons_making_compton = []
        for i, p in enumerate(self.event['particles'][0]):
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
            for j, p in enumerate(self.event['particles'][0]):
                if p.parent_track_id() == TID and p.track_id() not in compton_electron_tids:
                    compton_electron_tids.append(p.track_id())
        #print(' compton_electron_tids: ', compton_electron_tids)
        
        # Get track IDs of photons which of photons which make compton scattering (note: in contrast to above, only photons from pi0 decays which make compton scattering are included here)
        tids_of_photons_with_primary_compton = []
        for i, TID in enumerate(compton_electron_tids):
            for j, p in enumerate(self.event['particles'][0]):
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
            for i, p in enumerate(self.event['particles'][0]):
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
            for j, p in enumerate(self.event['particles'][0]):
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
        self.output['photonShowers'] = photon_showers
        
        # Assign the photon induced shower's properties (for showers where the initiating photon makes Compton Scattering)
        counter = 0
        for i, TID in enumerate(tids_of_photons_with_primary_compton):
            for j, p in enumerate(self.event['particles'][0]):
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
        self.output['comptonShowers'] = compton_showers
        
        return
    
    
    def extract_reco_information(self, event):
        '''
        Obtain reconstructed informations about pi0s and gammas originated from pi0 decays and dump
        it to self.reco_info['<variable>']
        '''

        self.reco_info['ev_id']                  = self.event['index']              # [-]
        self.reco_info['n_pi0']                  = len(self.output['matches'])      # [-]
        self.reco_info['n_gammas']               = 2.*len(self.output['matches'])   # [-]
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
        self.reco_info['OOFV']                   = self.output['OOFV']              # [-]

        showers = self.output['showers']
        
        # Note: match = if two showers point to the same point and this point is close to a track
        for match in range(self.reco_info['n_pi0']):
            match_1 = self.output['matches'][match][0]
            match_2 = self.output['matches'][match][1]               
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
            mask = self.output['shower_mask']                           # mask for all edeps classified as shower
            voxels_1 = self.output['showers'][match_1].voxels           # indices in the mask for the 1st match
            voxels_2 = self.output['showers'][match_2].voxels           # indices in the mask for the 2nd match
            edeps_1 = self.output['energy'][mask][voxels_1]             # all edeps for the 1st match
            edeps_2 = self.output['energy'][mask][voxels_2]             # all edeps for the 2nd match
            self.reco_info['gamma_voxels'].append(np.array(edeps_1))
            self.reco_info['gamma_voxels'].append(np.array(edeps_2))
            self.reco_info['gamma_n_voxels'].append(len(edeps_1))
            self.reco_info['gamma_n_voxels'].append(len(edeps_2))

        # Reconstructed angle and pi0 mass
        for match in self.output['matches']:
            idx1, idx2 = match
            s1, s2 = self.output['showers'][idx1], self.output['showers'][idx2]
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
