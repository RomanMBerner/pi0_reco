import numpy as np
from .utils import gamma_direction, cone_clusterer, pi0_pi_selection
from mlreco.utils.utils import CSVData

# Class that contains all the shower information
class Shower():
    def __init__(self, start=[], direction=[], voxels=[], energy=-1., pid=-1):
        self.start = start
        self.direction = direction
        self.voxels = voxels
        self.energy = energy
        self.pid = pid

# Chain object class that loads and stores the chain parameters
class Pi0Chain():

    def __init__(self, cfg, verbose=False):
        '''
        Initializes the chain from the configuration file
        '''
        self.cfg = cfg
        log_path = cfg['name']+'_log.csv'
        print('Initialized Pi0 mass chain, log path:', log_path)
        self._log = CSVData(log_path)
        self._keys = ['event_id', 'pion_id', 'pion_mass']
        self.verbose = verbose

    def log(self, eid, pion_id, pion_mass):
        self._log.record(self._keys, [eid, pion_id, pion_mass])
        self._log.write()
        self._log.flush()

    def run(self, event, event_id=0):
        '''
        Runs the full Pi0 reconstruction chain, from 3D charge
        information to Pi0 masses for events that contain one or more.
        '''
        # Filter out ghosts
        self.filter_ghosts(event)

        # Reconstruct energy
        self.reconstruct_energy(event)

        # Identify shower starting points
        self.find_shower_starts(event)
        if not len(event['showers']):
            if self.verbose:
                print('No shower start point found in event', event_id)
            return []

        # Reconstruct shower direction vectors
        self.reconstruct_shower_directions(event)

        # Reconstruct shower energy
        self.reconstruct_shower_energy(event)

        # Identify pi0 decays
        self.identify_pi0(event)
        if not len(event['matches']):
            if self.verbose:
                print('No pi0 found in event', event_id)
            return []

        # Compute masses
        masses = self.pi0_mass(event)

        # Log masses
        for i, m in enumerate(masses):
            self.log(event_id, i, m)

    def filter_ghosts(self, event):
        '''
        Removes ghost points from the charge tensor
        '''
        if self.cfg['input'] == 'energy':
            # No ghost to filter out
            pass

        elif self.cfg['segment']['method'] == 'mask':
            mask = np.where(event['ghost'][:,4] == 0) # TODO, should use 6-types
            event['charge'] = event['charge'][mask]
            # TODO must apply masks to more stuff, waiting for data

        elif self.cfg['segment']['method'] == 'uresnet':
            # Initialize the network
            raise NotImplementedError('GhostNet not implemented yet') # TODO, remove when it works
            from mlreco.main_funcs import process_config, prepare
            cfg = self.cfg['segment']['cfg']
            process_config(cfg)
            handlers = prepare(cfg)

            # Pass a batch through it (here size 1), unwrap output
            res = handlers.trainer.forward(event) # TODO will not work as is!

            # Argmax to determine most probable label
            pred_labels = np.argmax(res['segmentation'], axis=1)
            mask = np.where(pred_labels == 0) # TODO, should use 6-types
            event['charge'] = event['charge']['mask']
            # TODO must apply masks to more stuff, waiting for data

        else:
            raise ValueError('Ghost removal method not recognized:', self.cfg['ghost']['method'])

    def reconstruct_energy(self, event):
        '''
        Reconstructs energy deposition from charge
        '''
        if self.cfg['input'] == 'energy':
            # Energy already true
            pass

        elif self.cfg['response']['method'] == 'constant':
            reco = self.cfg['response']['factor']*event['charge'][:,4]
            event['energy'] = event['charge']
            event['energy'][:,4] = reco

        elif self.cfg['response']['method'] == 'enet':
            raise NotImplementedError('ENet not implemented yet')

        else:
            raise ValueError('Energy reconstruction method not recognized:', self.cfg['response']['method'])

    @staticmethod
    def is_shower(particle):
        '''
        Check if the particle is a shower
        '''
        pdg_code = abs(particle.pdg_code())
        if pdg_code == 22 or pdg_code == 11:
            return True
        return False

    def find_shower_starts(self, event):
        '''
        Identify starting points of showers
        '''
        if self.cfg['shower_start']['method'] == 'truth':
            # Get the true shower starting points from the particle information
            event['showers'] = []
            for i, part in enumerate(event['particles'][0]):
                if self.is_shower(part):
                    new_shower = Shower(start=[part.first_step().x(), part.first_step().y(), part.first_step().z()], pid=i)
                    event['showers'].append(new_shower)

        elif self.cfg['shower_start']['method'] == 'ppn':
            raise NotImplementedError('PPN not implemented yet')

        else:
            raise ValueError('EM shower primary identifiation method not recognized:', self.cfg['shower_start']['method'])

    def reconstruct_shower_directions(self, event):
        '''
        Reconstructs the direction of the showers
        '''
        if self.cfg['shower_dir']['method'] == 'truth':
            for shower in event['showers']:
                part = event['particles'][0][shower.pid]
                mom = [part.px(), part.py(), part.pz()]
                shower.direction = list(np.array(mom)/np.linalg.norm(mom))

        elif self.cfg['shower_dir']['method'] == 'pca':
            # Apply DBSCAN, PCA on the touching cluster to get angles
            points = np.array([s.start+[0,s.pid] for s in event['showers']])
            res, _, _ = gamma_direction.do_calculation(event['segment_label'], points)
            for i, shower in enumerate(event['showers']):
                shower.direction = list(res[i][-3:]/np.linalg.norm(res[i][-3:]))

        else:
            raise ValueError('Shower direction reconstruction method not recognized:', self.cfg['shower_dir']['method'])

    def reconstruct_shower_energy(self, event):
        '''
        Clusters the different showers, reconstruct energy of each shower
        '''
        if self.cfg['shower_energy']['method'] == 'truth':
            # Gets the true energy information from Geant4
            for shower in event['showers']:
                part = event['particles'][0][shower.pid]
                shower.energy = part.energy_init()
                pid = shower.pid
                #mask = np.where(event['group_label_full'][:,-2] == pid)[0]
                mask = np.where(event['group_label'][:,-1] == pid)[0]
                shower.voxels = mask

        elif self.cfg['shower_energy']['method'] == 'group':
            # Gets all the voxels in the group corresponding to the pid, adds up energy
            for shower in event['showers']:
                pid = shower.pid
                #mask = np.where(event['group_label_full'][:,-2] == pid)[0]
                mask = np.where(event['group_label'][:,-1] == pid)[0]
                shower.voxels = mask
                shower.energy = np.sum(event['energy'][mask][:,-1])

        elif self.cfg['shower_energy']['method'] == 'cone':
            # Fits cones to each shower, adds energies within that cone
            points = np.array([s.start+[0,s.pid] for s in event['showers']])
            res = cone_clusterer.find_shower_cone(event['dbscan_label'],
                event['group_label'], points, event['energy'],
                event['segment_label']) # This returns one array of voxel ids per primary
            for i, shower in enumerate(event['showers']):
                if not len(res[i]):
                    shower.energy = 0.
                    continue
                shower.voxels = res[i]
                shower.energy = np.sum(event['energy'][res[i]][:,4])

        else:
            raise ValueError('Shower energy reconstruction method not recognized:', self.cfg['shower_energy']['method'])

    def identify_pi0(self, event):
        '''
        Proposes pi0 candidates (match two showers)
        '''
        event['matches'] = []
        event['vertices'] = []
        n_showers = len(event['showers'])
        if self.cfg['shower_match']['method'] == 'truth':
            # Get the creation point of each particle. If two gammas originate from the same point,
            # It is most likely a pi0 decay.
            creations = []
            for shower in event['showers']:
                part = event['particles'][0][shower.pid]
                creations.append([part.position().x(), part.position().y(), part.position().z()])

            for i, ci in enumerate(creations):
                for j in range(i+1,n_showers):
                    if (np.array(ci) == np.array(creations[j])).all():
                        event['matches'].append([i,j])
                        event['vertices'].append(ci)

            return event['matches']

        elif self.cfg['shower_match']['method'] == 'proximity':
            # Check that shower direction vector interesect, within some tolerence
            for i in range(n_showers):
                gammai = np.array(event['showers'][i].start+[0]+event['showers'][i].direction)
                for j in range(i+1,n_showers):
                    gammaj = np.array(event['showers'][j].start+[0]+event['showers'][j].direction)
                    res = pi0_pi_selection.do_selection(event['segment_label'], gammai, gammaj)
                    if res[4] <= 10: # TODO, default return not ideal...
                        event['matches'].append([i,j])
                        event['vertices'].append(res[:3])

        else:
            raise ValueError('Shower matching method not recognized:', self.cfg['shower_match']['method'])

    @staticmethod
    def pi0_mass(event):
        '''
        Reconstructs the pi0 mass
        '''
        from math import sqrt
        masses = []
        for match in event['matches']:
            s1, s2 = event['showers'][match[0]], event['showers'][match[1]]
            e1, e2 = s1.energy, s2.energy
            t1, t2 = s1.direction, s2.direction
            costheta = np.dot(t1, t2)
            masses.append(sqrt(2*e1*e2*(1-costheta)))
        return masses

    def draw(self, event):
        from mlreco.visualization.voxels import scatter_voxels, scatter_label
        from plotly.offline import iplot
        import plotly.graph_objects as go

        # Create labels for the voxels
        # Use a different color for each cluster
        labels = np.full(len(event['energy'][:,4]), -1)
        for i, s in enumerate(event['showers']):
            labels[s.voxels] = i

        # Draw voxels with cluster labels
        voxels = event['energy'][:,:3]
        graph_voxels = scatter_label(voxels, labels, 2)[0]
        graph_voxels.name = 'Shower ID'
        graph_data = [graph_voxels]

        # Add EM primary points
        points = np.array([s.start for s in event['showers']])
        graph_start = scatter_voxels(points)[0]
        graph_start.name = 'Shower starts'
        graph_data.append(graph_start)

        # Add a vertex if matches, join vertex to start points
        for i, m in enumerate(event['matches']):
            v = event['vertices'][i]
            s1, s2 = event['showers'][m[0]].start, event['showers'][m[1]].start
            points = [v, s1, v, s2]
            line = scatter_voxels(np.array(points))[0]
            line.name = 'Pi0 Decay'
            line.mode = 'lines,markers'
            graph_data.append(line)

        # Draw
        iplot(graph_data)
