import numpy as np
import yaml
from copy import copy
from .directions.estimator import FragmentEstimator, DirectionEstimator
from .cluster.cone_clusterer import ConeClusterer
from .identification.matcher import Pi0Matcher
from mlreco.main_funcs import process_config, prepare
from mlreco.utils import CSVData
from mlreco.utils.ppn import uresnet_ppn_type_point_selector

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

    def __init__(self, io_cfg, chain_cfg, verbose=False):
        '''
        Initializes the chain from the configuration file
        '''
        # Initialize the data loader
        io_cfg = yaml.load(io_cfg,Loader=yaml.Loader)

        # Save config, initialize output
        self.cfg = chain_cfg
        self.verbose = verbose
        self.event = None
        self.output = {}

        # Initialize log
        log_path = chain_cfg['name']+'_log.csv'
        print('Initialized Pi0 mass chain, log path:', log_path)
        self._log = CSVData(log_path)
        self._keys = ['event_id', 'pion_id', 'pion_mass']

        # If a network is specified, initialize the network
        self.network = False
        if chain_cfg['segment'] == 'uresnet' or chain_cfg['shower_start'] == 'ppn':
            self.network = True
            with open(chain_cfg['net_cfg']) as cfg_file:
                net_cfg = yaml.load(cfg_file,Loader=yaml.Loader)
            io_cfg['model'] = net_cfg['model']
            io_cfg['trainval'] = net_cfg['trainval']
            
        # Initialize the fragment identifier
        self.frag_est = FragmentEstimator()
            
        # If a direction estimator is requested, initialize it
        if chain_cfg['shower_dir'] != 'truth':
            self.dir_est = DirectionEstimator()
            
        # If a clusterer is requested, initialize it
        if chain_cfg['shower_energy'] == 'cone':
            self.clusterer = ConeClusterer()
            
        # If a pi0 identifier is requested, initialize it
        if chain_cfg['shower_match'] == 'proximity':
            self.matcher = Pi0Matcher()

        # Pre-process configuration
        process_config(io_cfg)

        # Instantiate "handlers" (IO tools)
        self.hs = prepare(io_cfg)
        self.data_set = iter(self.hs.data_io)

    def hs(self):
        return self.hs

    def data_set(self):
        return self.data_set

    def log(self, eid, pion_id, pion_mass):
        self._log.record(self._keys, [eid, pion_id, pion_mass])
        self._log.write()
        self._log.flush()

    def run(self):
        '''
        Runs the full Pi0 reconstruction chain, from 3D charge
        information to Pi0 masses for events that contain one
        or more Pi0 decay.
        '''
        n_events = len(self.hs.data_io)
        for i in range(n_events):
            self.run_loop()

    def run_loop(self):
        '''
        Runs the full Pi0 reconstruction chain on a single event,
        from 3D charge information to Pi0 masses for events that
        contain one or more Pi0 decay.
        '''
        # Reset output
        self.output = {}
        
        # Load data
        if not self.network:
            event = next(self.data_set)
            event_id = event['index'][0]
        else:
            event, self.output['forward'] = self.hs.trainer.forward(self.data_set)
            for key in event.keys():
                if key != 'particles':
                    event[key] = event[key][0]
            event_id = event['index']

        self.event = event

        # Filter out ghosts
        self.filter_ghosts(event)

        # Reconstruct energy
        self.reconstruct_energy(event)

        # Identify shower starting points, skip if there is less than 2 (no pi0)
        self.find_shower_starts(event)
        if len(self.output['showers']) < 2:
            if self.verbose:
                print('No shower start point found in event', event_id)
            return []
        
        # Match primary shower fragments with each start points
        if self.cfg['shower_dir'] != 'truth' or self.cfg['shower_energy'] == 'cone':
            self.match_primary_fragments(event)
            if not len(self.output['fragments']):
                if self.verbose:
                    print('Could not find a fragment for each start point in event', event_id)
                return []

        # Reconstruct shower direction vectors
        self.reconstruct_shower_directions(event)

        # Reconstruct shower energy
        self.reconstruct_shower_energy(event)

        # Identify pi0 decays
        self.identify_pi0(event)
        if not len(self.output['matches']):
            if self.verbose:
                print('No pi0 found in event', event_id)
            return []

        # Compute masses
        masses = self.pi0_mass()

        # Log masses
        for i, m in enumerate(masses):
            self.log(event_id, i, m)

    def filter_ghosts(self, event):
        '''
        Removes ghost points from the charge tensor
        '''
        if self.cfg['input'] == 'energy':
            self.output['segment'] = event['segment_label_true']
            self.output['group'] = event['group_label_true']
            self.output['dbscan'] = event['dbscan_label_true']

        elif self.cfg['segment'] == 'mask':
            mask = np.where(event['segment_label_reco'][:,-1] != 5)[0]
            self.output['charge'] = event['charge'][mask]
            self.output['segment'] = event['segment_label_reco'][mask]
            self.output['group'] = event['group_label_reco'] # group_label_reco is wrong, so no masking, TODO
            self.output['dbscan'] = event['dbscan_label_reco'][mask]

        elif self.cfg['segment'] == 'uresnet':
            # Get the segmentation output of the network
            res = self.output['forward']

            # Argmax to determine most probable label
            pred_ghost = np.argmax(res['ghost'][0], axis=1)
            pred_labels = np.argmax(res['segmentation'][0], axis=1)
            mask = np.where(pred_ghost == 0)[0]
            self.output['charge'] = event['charge'][mask]
            self.output['segment'] = copy(event['segment_label_reco'])
            self.output['segment'][:,-1] = pred_labels
            self.output['segment'] = self.output['segment'][mask]
            self.output['group'] = event['group_label_reco'] # group_label_reco is wrong, so no masking, TODO
            self.output['dbscan'] = event['dbscan_label_reco'][mask]

        else:
            raise ValueError('Semantic segmentation method not recognized:', self.cfg['segment'])

    def reconstruct_energy(self, event):
        '''
        Reconstructs energy deposition from charge
        '''
        if self.cfg['input'] == 'energy':
            self.output['energy'] = event['energy']

        elif self.cfg['response'] == 'constant':
            reco = self.cfg['response_cst']*self.output['charge'][:,-1]
            self.output['energy'] = copy(self.output['charge'])
            self.output['energy'][:,-1] = reco

        elif self.cfg['response'] == 'average':
            self.output['energy'] = copy(self.output['charge'])
            self.output['energy'][:,-1] = self.cfg['response_average']

        elif self.cfg['response'] == 'full':
            raise NotImplementedError('Proper energy reconstruction not implemented yet')

        elif self.cfg['response'] == 'enet':
            raise NotImplementedError('ENet not implemented yet')

        else:
            raise ValueError('Energy reconstruction method not recognized:', self.cfg['response'])

    def find_shower_starts(self, event):
        '''
        Identify starting points of showers
        '''
        if self.cfg['shower_start'] == 'truth':
            # Get the true shower starting points from the particle information
            self.output['showers'] = []
            for i, part in enumerate(event['particles'][0]):
                if self.is_shower(part):
                    new_shower = Shower(start=[part.first_step().x(), part.first_step().y(), part.first_step().z()], pid=i)
                    self.output['showers'].append(new_shower)

        elif self.cfg['shower_start'] == 'ppn':
            # Parse the PPN output as a list of points, select EM point only
            ppn = uresnet_ppn_type_point_selector(self.event['input_data'], self.output['forward'], entry=0, score_threshold=0.5, type_threshold=5, eps=4.99)
            self.output['showers'] = []
            for i, p in enumerate(ppn):
                if p[-1] == 2:
                    new_shower = Shower(start=p[:3], pid=i)
                    self.output['showers'].append(new_shower)

        else:
            raise ValueError('EM shower primary identifiation method not recognized:', self.cfg['shower_start'])
            
    def match_primary_fragments(self, event):
        '''
        For each shower start point, find the closest DBSCAN shower cluster
        '''
        # Mask out points that are not showers
        shower_mask = np.where(self.output['segment'][:,-1] == 2)[0]
        if not len(shower_mask):
            self.output['fragments'] = []
            return
        
        # Assign clusters
        points = np.array([s.start for s in self.output['showers']])
        clusts = self.frag_est.assign_frags_to_primary(self.output['energy'][shower_mask], points)
        if len(clusts) != len(points):
            self.output['fragments'] = []
            return
        
        # Return list of voxel indices for each cluster
        self.output['fragments'] = clusts

    def reconstruct_shower_directions(self, event):
        '''
        Reconstructs the direction of the showers
        '''
        if self.cfg['shower_dir'] == 'truth':
            for shower in self.output['showers']:
                part = event['particles'][0][shower.pid]
                mom = [part.px(), part.py(), part.pz()]
                shower.direction = list(np.array(mom)/np.linalg.norm(mom))

        elif self.cfg['shower_dir'] == 'pca' or self.cfg['shower_dir'] == 'cent':
            # Apply DBSCAN, PCA on the touching cluster to get angles
            algo = self.cfg['shower_dir']
            mask = np.where(self.output['segment'][:,-1] == 2)[0]
            points = np.array([s.start for s in self.output['showers']])
            try:
                res = self.dir_est.get_directions(self.output['energy'][mask], 
                    points, self.output['fragments'], max_distance=float('inf'), mode=algo)
            except AssertionError as err: # Cluster was not found for at least one primary
                if self.verbose:
                    print('Error in direction reconstruction:', err)
                res = [[0., 0., 0.] for _ in range(len(self.output['showers']))]
                    
            for i, shower in enumerate(self.output['showers']):
                shower.direction = res[i]

        else:
            raise ValueError('Shower direction reconstruction method not recognized:', self.cfg['shower_dir'])

    def reconstruct_shower_energy(self, event):
        '''
        Clusters the different showers, reconstruct energy of each shower
        '''
        if self.cfg['shower_energy'] == 'truth':
            # Gets the true energy information from Geant4
            for shower in self.output['showers']:
                part = event['particles'][0][shower.pid]
                shower.energy = part.energy_init()
                pid = shower.pid
                mask = np.where(self.output['group'][:,-1] == pid)[0]
                shower.voxels = mask

        elif self.cfg['shower_energy'] == 'group':
            # Gets all the voxels in the group corresponding to the pid, adds up energy
            for shower in self.output['showers']:
                pid = shower.pid
                mask = np.where(self.output['group'][:,-1] == pid)[0]
                shower.voxels = mask
                shower.energy = np.sum(self.output['energy'][mask][:,-1])

        elif self.cfg['shower_energy'] == 'cone':
            # Fits cones to each shower, adds energies within that cone
            points = np.array([s.start for s in self.output['showers']])
            dirs = np.array([s.direction for s in self.output['showers']])
            mask = np.where(self.output['segment'][:,-1] == 2)[0]
            try:
                pred = self.clusterer.fit_predict(self.output['energy'][mask,:3], points, self.output['fragments'], dirs)
            except (ValueError, AssertionError):
                for i, shower in enumerate(self.output['showers']):
                    shower.voxels = []
                    shower.energy = 0.
                return
            padded_pred = np.full(len(self.output['segment']), -1)
            padded_pred[mask] = pred
            for i, shower in enumerate(self.output['showers']):
                shower_mask = np.where(padded_pred == i)[0]
                if not len(shower_mask):
                    shower.energy = 0.
                    continue
                shower.voxels = shower_mask
                shower.energy = np.sum(self.output['energy'][shower_mask][:,-1])

        else:
            raise ValueError('Shower energy reconstruction method not recognized:', self.cfg['shower_energy'])

    def identify_pi0(self, event):
        '''
        Proposes pi0 candidates (match two showers)
        '''
        self.output['matches'] = []
        self.output['vertices'] = []
        n_showers = len(self.output['showers'])
        if self.cfg['shower_match'] == 'truth':
            # Get the creation point of each particle. If two gammas originate from the same point,
            # It is most likely a pi0 decay.
            creations = []
            for shower in self.output['showers']:
                part = event['particles'][0][shower.pid]
                creations.append([part.position().x(), part.position().y(), part.position().z()])

            for i, ci in enumerate(creations):
                for j in range(i+1,n_showers):
                    if (np.array(ci) == np.array(creations[j])).all():
                        self.output['matches'].append([i,j])
                        self.output['vertices'].append(ci)

            return self.output['matches']

        elif self.cfg['shower_match'] == 'proximity':
            # Pair closest shower vectors
            points = np.array([s.start for s in self.output['showers']])
            dirs = np.array([s.direction for s in self.output['showers']])
            try:
                self.output['matches'], self.output['vertices'], dists =\
                    self.matcher.find_matches(points, dirs, self.output['segment'])
            except ValueError as err:
                if self.verbose:
                    print('Error in PID:', err)
                return

            if self.cfg['refit_dir']:
                for i, m in enumerate(self.output['matches']):
                    v = np.array(self.output['vertices'][i])
                    for j in m:
                        new_dir = np.array(points[j]) - v
                        self.output['showers'][j].direction = new_dir/np.linalg.norm(new_dir)

            if self.cfg['shower_energy'] == 'cone' and self.cfg['refit_cone']:
                self.reconstruct_shower_energy(event)

        else:
            raise ValueError('Shower matching method not recognized:', self.cfg['shower_match'])

    def pi0_mass(self):
        '''
        Reconstructs the pi0 mass
        '''
        from math import sqrt
        masses = []
        for match in self.output['matches']:
            s1, s2 = self.output['showers'][match[0]], self.output['showers'][match[1]]
            e1, e2 = s1.energy, s2.energy
            t1, t2 = s1.direction, s2.direction
            costheta = np.dot(t1, t2)
            if abs(costheta) > 1.:
                masses.append(0.)
                continue
            masses.append(sqrt(2*e1*e2*(1-costheta)))
        self.output['masses'] = masses
        return masses

    def draw(self, draw_input=False):
        from mlreco.visualization import plotly_layout3d
        from mlreco.visualization.voxels import scatter_voxels, scatter_label
        import plotly.graph_objs as go
        from plotly.offline import iplot

        # If requested, draw the input of the chain
        if draw_input:
            if self.cfg['input'] == 'energy':
                graph_input = scatter_label(self.event['energy'], self.event['energy'][:,-1], 2)
            else:
                graph_input = scatter_label(self.event['charge'], self.event['charge'][:,-1], 2)

            iplot(go.Figure(data=graph_input, layout=plotly_layout3d()))

        # Create labels for the voxels
        # Use a different color for each cluster
        labels = np.full(len(self.output['energy'][:,-1]), -1)
        for i, s in enumerate(self.output['showers']):
            labels[s.voxels] = i

        # Draw voxels with cluster labels
        voxels = self.output['energy'][:,:3]
        graph_voxels = scatter_label(voxels, labels, 2)[0]
        graph_voxels.name = 'Shower ID'
        graph_data = [graph_voxels]

        if len(self.output['showers']):
            # Add EM primary points
            points = np.array([s.start for s in self.output['showers']])
            graph_start = scatter_voxels(points)[0]
            graph_start.name = 'Shower starts'
            graph_data.append(graph_start)
            
            # Add EM primary directions
            dirs = np.array([s.direction for s in self.output['showers']])
            arrows = go.Cone(x=points[:,0], y=points[:,1], z=points[:,2], 
                             u=dirs[:,0], v=dirs[:,1], w=dirs[:,2],
                             sizemode='absolute', sizeref=0.25, anchor='tail',
                             showscale=False, opacity=0.4)
            graph_data.append(arrows)

            # Add a vertex if matches, join vertex to start points
            for i, m in enumerate(self.output['matches']):
                v = self.output['vertices'][i]
                s1, s2 = self.output['showers'][m[0]].start, self.output['showers'][m[1]].start
                points = [v, s1, v, s2]
                line = scatter_voxels(np.array(points))[0]
                line.name = 'Pi0 Decay'
                line.mode = 'lines,markers'
                graph_data.append(line)

        # Draw
        iplot(go.Figure(data=graph_data,layout=plotly_layout3d()))

    @staticmethod
    def is_shower(particle):
        '''
        Check if the particle is a shower
        '''
        pdg_code = abs(particle.pdg_code())
        if pdg_code == 22 or pdg_code == 11:
            return True
        return False
