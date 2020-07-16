import numpy as np
from numpy import linalg
import yaml
import torch
from copy import copy
from larcv import larcv
from scipy.spatial import distance
from .directions.estimator import FragmentEstimator, DirectionEstimator
from .cluster.start_finder import StartPointFinder
from .cluster.cone_clusterer import ConeClusterer
from .cluster.dbscan import DBSCANCluster
from .identification.matcher import Pi0Matcher
from mlreco.main_funcs import process_config, prepare
from mlreco.utils import CSVData
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.gnn.cluster import cluster_direction

# Class that contains all the shower information
class Shower():
    def __init__(self, start=-np.ones(3), direction=-np.ones(3), voxels=[], energy=-1., pid=-1, group_pred=-1):
        self.start = start
        self.direction = direction
        self.voxels = voxels
        self.energy = energy
        self.pid = int(pid)
        self.group_pred = int(group_pred)
        
    def __str__(self):
        return """
        Shower ID  : {}
        Start point: ({:0.2f},{:0.2f},{:0.2f})
        Direction  : ({:0.2f},{:0.2f},{:0.2f})
        Voxel count: {}
        Energy     : {}
        Group_pred : {}
        """.format(self.pid, *self.start, *self.direction, len(self.voxels), self.energy, self.group_pred)

class ElectronShower():
    def __init__(self, pid=-1, group_id=-1, start=-np.ones(3), mom_init=-np.ones(3), direction=-np.ones(3), voxels=[], einit=-1, edeps=[], edep_tot=-1):
        self.pid        = int(pid)
        self.group_id   = int(group_id)
        self.start      = start
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
        Start      : ({:0.2f},{:0.2f},{:0.2f})
        Mom_init   : ({:0.2f},{:0.2f},{:0.2f})
        Direction  : ({:0.2f},{:0.2f},{:0.2f})
        Voxels     : {}
        Voxel count: {}
        E init     : {}
        E deposits : {}
        E dep tot  : {}
        """.format(self.pid, self.group_id, *self.start, *self.mom_init, *self.direction, self.voxels, self.n_voxels, self.einit, self.edeps, self.edep_tot)
    
class PhotonShower():
    def __init__(self, pid=-1, group_id=-1, start=-np.ones(3), mom_init=-np.ones(3), direction=-np.ones(3), voxels=[], einit=-1, edeps=[], edep_tot=-1):
        self.pid        = int(pid)
        self.group_id   = int(group_id)
        self.start      = start
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
        Start      : ({:0.2f},{:0.2f},{:0.2f})
        Mom_init   : ({:0.2f},{:0.2f},{:0.2f})
        Direction  : ({:0.2f},{:0.2f},{:0.2f})
        Voxels     : {}
        Voxel count: {}
        E init     : {}
        E deposits : {}
        E dep tot  : {}
        """.format(self.pid, self.group_id, *self.start, *self.mom_init, *self.direction, self.voxels, self.n_voxels, self.einit, self.edeps, self.edep_tot)

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



# Chain object class that loads and stores the chain parameters
class Pi0Chain():

    #Class constants
    IDX_SEMANTIC_ID = -1
    IDX_GROUP_ID = -2
    IDX_CLUSTER_ID = -3


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
        self.true_info = {}
        self.reco_info = {}

        # Initialize log
        #log_path = chain_cfg['name']+'_log.csv'
        log_path = 'masses_fiducialized_' + str(chain_cfg['fiducialize']) + 'px.csv'
        print('Initialized Pi0 mass chain, log path:', log_path)
        self._log = CSVData(log_path)
        self._keys = ['event_id', 'pion_id', 'pion_mass']

        # If a network is specified, initialize the network
        self.network = False
        if chain_cfg['segment'] == 'uresnet' or chain_cfg['shower_start'] == 'ppn' or chain_cfg['shower_start'] == 'gnn':
            self.network = True
            with open(chain_cfg['net_cfg']) as cfg_file:
                net_cfg = yaml.load(cfg_file,Loader=yaml.Loader)
            io_cfg['model'] = net_cfg['model']
            io_cfg['trainval'] = net_cfg['trainval']

        # Initialize the fragment identifier
        self.frag_est = DBSCANCluster()

        # If a direction estimator is requested, initialize it
        if chain_cfg['shower_direction'] != 'label':
            self.dir_est = DirectionEstimator()

        # If a clusterer is requested, initialize it
        if chain_cfg['shower_cluster'] in ['cone', 'gnn']:
            self.clusterer = ConeClusterer()

        # If a pi0 identifier is requested, initialize it
        if (chain_cfg['shower_match'] == 'proximity' or chain_cfg['shower_match'] == 'ppn'):
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


    def select_overlap(self, a0, a1, overlap=True, dim=3):
        '''
        Given 2 arrays of shape (N,dim+) where dim indicates x,y,z voxel coordinates,
        compute the common points across 2 arrays and return index that corresponds to a0
        '''
        coords0 = a0[:,:dim].astype(np.int16)
        coords1 = a1[:,:dim].astype(np.int16)
        if overlap:
            idx = [ i for i in range(len(coords0)) if (coords0[i] == coords1).all(axis=1).any() ]
        else:
            idx = [ i for i in range(len(coords0)) if not (coords0[i] == coords1).all(axis=1).any() ]
        return idx


    def layout(self, width=1024, height=768, xrange=(0,768), yrange=(0,768), zrange=(0,768), dark=False, aspectmode='cube' ):
        import plotly.graph_objs as go

        layout = go.Layout(
            showlegend=True,
            legend=dict(x=1.01,y=0.95),
            width=width,
            height=height,
            hovermode='closest',
            margin=dict(l=0,r=0,b=0,t=0),
            #template='plotly_dark',
            uirevision = 'same',
            scene = dict(xaxis = dict(nticks=10, range = xrange, showticklabels=True, title='x'),
                         yaxis = dict(nticks=10, range = yrange, showticklabels=True, title='y'),
                         zaxis = dict(nticks=10, range = zrange, showticklabels=True, title='z'),
                         aspectmode=aspectmode)
        )
        if dark: layout.template = 'plotly_dark'
        return layout


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
        
        # Initialize some objects for reco_info
        # TODO: Check where it would it fit best
        self.reco_info['n_pi0']                  = 0    # [-]
        self.reco_info['n_gammas']               = 0    # [-]
        self.reco_info['matches']                = []   # [-]
        self.reco_info['gamma_mom']              = []   # [MeV/c]
        self.reco_info['gamma_dir']              = []   # [x,y,z]
        self.reco_info['gamma_start']            = []   # [x,y,z] # pi0->2gamma vertex
        self.reco_info['gamma_edep']             = []   # [MeV]
        self.reco_info['gamma_pid']              = []   # [-]
        self.reco_info['gamma_voxels_mask']      = []   # [-]
        self.reco_info['gamma_n_voxels_mask']    = []   # [-]
        self.reco_info['gamma_voxels']           = []   # [-]
        self.reco_info['gamma_n_voxels']         = []   # [-]
        self.reco_info['gamma_angle']            = []   # [rad]
        self.reco_info['pi0_mass']               = []   # [MeV/c2]
        self.reco_info['OOFV']                   = []   # [-]
        
        # Check input
        self.infer_inputs(event)

        # Set the semantics
        self.infer_semantics(event)

        # Extract true information about pi0 -> gamma + gamma
        self.extract_true_information(event)
        
        # Find showers induced by (primary) electrons
        self.find_true_electron_showers(event)
        
        # Find showers induced by photons from a pi0 decay
        self.find_true_photon_showers(event)

        # Filter out ghosts
        self.filter_ghosts(event)
        
        # Obtain points from PPN
        if (self.cfg['shower_start'] == 'ppn' or self.cfg['shower_match'] == 'ppn'):
            self.obtain_ppn_points(event)

        # Reconstruct energy
        self.charge_to_energy(event)

        # Check data dimensions
        assert self.output['energy' ].shape == self.output['charge'].shape
        assert self.output['energy' ].shape == self.output['segment'].shape

        # Identify shower starting points, skip if there are less than 2 (no pi0)
        self.reconstruct_shower_starts(event)
        if len(self.output['showers']) < 2:
            if self.verbose:
                print('< 2 shower start points found in event', event_id)
            return

        # Form shower fragments
        self.reconstruct_shower_fragments(event)
        if len(self.output['showers']) < 2:
            if self.verbose:
                print('< 2 shower fragment found in event', event_id)
            return

        # Reconstruct shower direction vectors
        self.reconstruct_shower_directions(event)

        # Reconstruct shower cluster
        self.reconstruct_shower_cluster(event)

        # Reconstruct shower energy
        self.reconstruct_shower_energy(event)

        # Identify pi0 decays
        self.identify_pi0(event)
        if not len(self.output['matches']):
            if self.verbose:
                print('No pi0 found in event', event_id)
            return

        # Make fiducialization (put shower number to self.output['OOFV'] if >0 edep of the shower is OOFV)
        # This is relatively strict -> might want to add the shower to OOFV only if a certain fraction of all edeps is OOFV
        # TODO: Check whether this function is still needed
        self.fiducialize(event)

        # Compute masses
        masses = self.pi0_mass()

        # Log masses
        for i, m in enumerate(masses):
            self.log(event_id, i, m)
        
        # Extract reco information about pi0 -> gamma + gamma
        self.extract_reco_information(event)


    def infer_inputs(self,event):
        if self.cfg['deghost'] == "label" or self.cfg['segment'] == 'label':
            assert 'segment_label' in event
        if self.cfg['charge2energy'] == 'label':
            assert 'energy_label' in event
            self.output['energy_label'] = copy(event['energy_label'])
        if 'label' in [ self.cfg['shower_fragment'], self.cfg['shower_direction'],
                        self.cfg['shower_cluster'], self.cfg['shower_energy'] ]:
            assert self.cfg['shower_start'] == 'label'
        if self.cfg['shower_start'] == 'label':
            assert 'particles' in event
        if self.cfg['shower_fragment'] == 'label' or self.cfg['shower_cluster'] == 'label':
            assert 'cluster_label' in event
            self.output['cluster_label'] = copy(event['cluster_label'])

        self.output['charge'] = copy(event['input_data'])

        assert not 'segment_label' in event or event['segment_label'].shape == event['input_data'].shape
        assert not 'energy_label'  in event or event['energy_label' ].shape == event['input_data'].shape


    def infer_semantics(self, event):
        if self.cfg['segment'] == 'label':
            self.output['segment'] = event['segment_label']

        elif self.cfg['segment'] == 'uresnet':
            # Get the segmentation output of the network
            res = self.output['forward']
            # Argmax to determine most probable label
            self.output['segment'] = copy(event['segment_label'])
            self.output['segment'][:,-1] = np.argmax(res['segmentation'][0], axis=1)
        else:
            raise ValueError('Semantic segmentation method not recognized:', self.cfg['segment'])

        self.output['shower_mask'] = np.where(self.output['segment'][:,-1] == larcv.kShapeShower)
        for tag in ['cluster_label']:
            if not tag in self.output: continue
            segment = self.output['segment']
            shower_segment = segment[self.output['shower_mask']]
            masked_label = []
            label = self.output[tag]
            for idx, particle in enumerate(event['particles'][0]):
                label_segment = label[np.where(label[:,self.IDX_CLUSTER_ID] == particle.id())]
                if not particle.shape() == larcv.kShapeShower:
                    masked_label.append(label_segment)
                    continue
                valid_idx = self.select_overlap(label_segment,shower_segment)
                masked_label.append(label_segment[valid_idx])
            self.output[tag] = np.concatenate(masked_label)


    def filter_ghosts(self, event):
        '''
        Removes ghost points from the charge tensor
        '''
        mask = None
        if self.cfg['deghost'] == 'label':
            mask = np.where(event['segment_label'][:,-1] != 5)[0]

        elif self.cfg['deghost'] == 'uresnet':
            # Get the segmentation output of the network
            res = self.output['forward']
            # Argmax to determine most probable label
            pred_ghost = np.argmax(res['ghost'][0], axis=1)
            mask = np.where(pred_ghost == 0)[0]

        elif self.cfg['deghost']:
            raise ValueError('De-ghosting method not recognized:', self.cfg['deghost'])

        else:
            # no de-ghosting needed: return!
            return

        if 'charge'  in self.output: self.output['charge' ] = self.output['charge' ][mask]
        if 'segment' in self.output: self.output['segment'] = self.output['segment'][mask]
        if 'energy_label' in self.output: self.output['energy_label'] = self.output['energy_label'][mask]

        for tag in ['cluster_label']:
            if not tag in self.output: continue
            segment = self.output['segment']
            shower_segment = segment[self.output['shower_mask']]
            masked_label = []
            label = self.output[tag]
            for idx, particle in enumerate(event['particles']):
                label_segment = label[np.where(label[:,self.IDX_CLUSTER_ID] == particle.id())]
                if not particle.shape() == larcv.kShapeShower:
                    masked_label.append(label_segment)
                    continue
                valid_idx = self.select_overlap(label_segment,shower_segment)
                masked_label.append(label_segment[valid_idx])
            self.output[tag] = np.concatenate(masked_label)


    def obtain_ppn_points(self, event):
        '''
        Obtain points predicted by PPN, for each semantic class
        '''
        from mlreco.utils.ppn import uresnet_ppn_type_point_selector
            
        point_score_index  = -1 * (int(larcv.kShapeUnknown) + 1)
        shower_score_index = -1 * (int(larcv.kShapeUnknown) - 1 - int(larcv.kShapeShower))
        track_score_index  = -1 * (int(larcv.kShapeUnknown) - 1 - int(larcv.kShapeTrack))
        michel_score_index = -1 * (int(larcv.kShapeUnknown) - 1 - int(larcv.kShapeMichel))
        delta_score_index  = -1 * (int(larcv.kShapeUnknown) - 1 - int(larcv.kShapeDelta))
        LEScat_score_index = -1 * (int(larcv.kShapeUnknown) - 1 - int(larcv.kShapeLEScatter))
            
        points = uresnet_ppn_type_point_selector([event['input_data']],self.output['forward'])
        
        default_scores = [0.9, 0.9, 0.9, 0.9, 0.9]
        
        try:
            thresholds = self.cfg['PPN_score_thresh']
            if not (len(thresholds)==5):
                print(' Need exactly 5 thresholds for PPN scores,', len(thresholds), 'given. Set them to', default_scores, ' ... ')
                thresholds = default_scores
        except:
            print(' Thresholds for PPN scores not defined in chain cfg. Set them to', default_scores, ' ... ')
            thresholds = default_scores

        shower_points = points[0][np.where(points[0][:,shower_score_index] > float(thresholds[0]))]
        track_points  = points[0][np.where(points[0][:,track_score_index]  > float(thresholds[1]))]
        michel_points = points[0][np.where(points[0][:,michel_score_index] > float(thresholds[2]))]
        delta_points  = points[0][np.where(points[0][:,delta_score_index]  > float(thresholds[3]))]
        LEScat_points = points[0][np.where(points[0][:,LEScat_score_index] > float(thresholds[4]))]
            
        shower_total_score = shower_points[:,shower_score_index] * shower_points[:,point_score_index]
        track_total_score  = track_points[:,track_score_index]   * track_points[:,point_score_index]
        michel_total_score = michel_points[:,michel_score_index] * michel_points[:,point_score_index]
        delta_total_score  = delta_points[:,delta_score_index]   * delta_points[:,point_score_index]
        LEScat_total_score = LEScat_points[:,LEScat_score_index] * LEScat_points[:,point_score_index]
            
        shower_ordered = np.argsort(shower_total_score)
        track_ordered  = np.argsort(track_total_score)
        michel_ordered = np.argsort(michel_total_score)
        delta_ordered  = np.argsort(delta_total_score)
        LEScat_ordered = np.argsort(LEScat_total_score)
            
        self.output['PPN_shower_points'] = [ShowerPoints(ppns=shower_points[i,:3],shower_score=i,shower_id=int(i)) for i in shower_ordered]
        self.output['PPN_track_points']  = [TrackPoints(ppns=track_points[i,:3],track_score=i,track_id=int(i)) for i in track_ordered]
        self.output['PPN_michel_points'] = [MichelPoints(ppns=michel_points[i,:3],michel_score=i,michel_id=int(i)) for i in michel_ordered]
        self.output['PPN_delta_points']  = [DeltaPoints(ppns=delta_points[i,:3],delta_score=i,delta_id=int(i)) for i in delta_ordered]
        self.output['PPN_LEScat_points'] = [LEScatPoints(ppns=LEScat_points[i,:3],LEScat_score=i,LEScat_id=int(i)) for i in LEScat_ordered]


    def charge_to_energy(self, event):
        '''
        Reconstructs energy deposition from charge
        '''
        if self.cfg['charge2energy'] is None:
            self.output['energy'] = copy(self.output['charge'])

        elif self.cfg['charge2energy'] == 'label':
            self.output['energy'] = self.output['energy_label']

        elif self.cfg['charge2energy'] == 'constant':
            reco = self.cfg['charge2energy_cst']*self.output['charge'][:,-1]
            self.output['energy'] = copy(self.output['charge'])
            self.output['energy'][:,-1] = reco

        elif self.cfg['charge2energy'] == 'average':
            self.output['energy'] = copy(self.output['charge'])
            self.output['energy'][:,-1] = self.cfg['charge2energy_average']

        elif self.cfg['charge2energy'] == 'full':
            raise NotImplementedError('Proper energy reconstruction not implemented yet')

        elif self.cfg['charge2energy'] == 'enet':
            raise NotImplementedError('ENet not implemented yet')

        else:
            raise ValueError('Energy reconstruction method not recognized:', self.cfg['charge2energy'])


    def reconstruct_shower_starts(self, event):
        '''
        Identify starting points of showers. Points should be ordered by the definiteness of a shower
        '''
        if self.cfg['shower_start'] == 'label':
            # Find showers
            particles = event['particles'][0]
            points = event['ppn_label']
            points = points[np.where(points[:,-2]==larcv.kShapeShower)]
            order  = np.argsort([particles[int(points[i,-1])].energy_deposit() for i in range(len(points))])
            if not self.cfg['shower_cluster'] == 'label':
                self.output['showers'] = [Shower(start=points[i,:3],pid=points[i,-1]) for i in order]
            else:
                # create a list of group labels
                primaries = {}
                for i,p in enumerate(particles):
                    if p.group_id() < 0: continue
                    if not p.shape() == larcv.kShapeShower: continue
                    if not p.group_id() in primaries: primaries[p.group_id()] = p
                    elif p.position().t() < primaries[p.group_id()].position().t():
                        primaries[p.group_id()] = p
                    #print(i,p.id(),p.shape(),p.pdg_code(),p.creation_process())
                showers = []
                for gid,p in primaries.items():
                    # find a point defined by ppn
                    start = points[points[:,-1] == p.id()]
                    assert len(start) < 2
                    if len(start) == 0:
                        print('Ignoring a true shower due to not finding PPN label point!')
                        print('ID =',p.id())
                        print(p.dump())
                        continue
                    showers.append(Shower(start=start[0,:3],pid=int(p.id())))
                self.output['showers'] = showers

        #elif self.cfg['shower_start'] == 'ppn':
        #    from mlreco.utils.ppn import uresnet_ppn_type_point_selector
        #    shower_score_index = -1 * (int(larcv.kShapeUnknown) - int(larcv.kShapeShower))
        #    point_score_index  = -1 * (int(larcv.kShapeUnknown) + 1)
        #    points = uresnet_ppn_type_point_selector([event['input_data']],self.output['forward'])
        #    #points = points[np.where(points[:,shower_score_index] > self.cfg.get('ppn_shower_score_thresh',0.5))]
        #    points = points[0][np.where(points[0][:,shower_score_index] > self.cfg.get('ppn_shower_score_thresh',0.5))]
        #    total_score = points[:,shower_score_index] * points[:,point_score_index]
        #    order  = np.argsort(total_score)
        #    self.output['showers'] = [Shower(start=points[i,:3],pid=int(i)) for i in order]

        elif (self.cfg['shower_start'] == 'gnn' or self.cfg['shower_start'] == 'ppn'):
            # Use the node predictions to find primary nodes
            if not 'node_pred' in self.output['forward']:
                self.output['showers'] = []
                return
            from scipy.special import softmax
            node_scores = softmax(self.output['forward']['node_pred'][0], axis=1)
            primary_labels = np.zeros(len(node_scores), dtype=bool)
            group_ids = self.output['forward']['group_pred'][0]
            for g in np.unique(group_ids):
                mask = np.where(group_ids == g)[0]
                idx  = node_scores[mask][:,1].argmax()
                primary_labels[mask[idx]] = True
            primaries = np.where(primary_labels)[0]
            primary_clusts = self.output['forward']['shower_fragments'][0][primaries]
            all_clusts = np.array([np.array(c) for c in self.output['forward']['shower_fragments'][0]])
            group_ids = self.output['forward']['group_pred'][0]
            
            # Obtain group_id predicted by GNN for all primary_clusts
            primary_clusts_group_pred = []
            for i, primary_cluster in enumerate(primary_clusts):
                for j, all_cluster in enumerate(all_clusts):
                    if all_cluster[0]==primary_cluster[0]:
                        primary_clusts_group_pred.append(group_ids[j])
                        break

            if self.cfg['shower_start'] == 'gnn': # Getting start points from GNN
                start_finder = StartPointFinder()
                start_points = start_finder.find_start_points(self.output['energy'][:,:3], primary_clusts)
            
            elif self.cfg['shower_start'] == 'ppn': # Getting start points from PPN
                start_points = []
                for clust in primary_clusts:
                    scores = softmax(self.output['forward']['points'][0][clust,3:5], axis=1)
                    argmax = (scores[:,-1]).argmax()
                    pos = self.output['energy'][clust][argmax,:3] + self.output['forward']['points'][0][clust][argmax,:3] + 0.5 # +0.5 (middle of voxel)
                    start_points.append(pos)
            else:
                raise ValueError('Shower_start method not recognized:', self.cfg['shower_start'])

            self.output['showers'] = [Shower(start=p,pid=int(i),group_pred=int(primary_clusts_group_pred[i])) for i, p in enumerate(start_points)]

        else:
            raise ValueError('EM shower primary identifiation method not recognized:', self.cfg['shower_start'])


    def reconstruct_shower_fragments(self,event):
        '''
        Cluster shower pixels (fragmentation) per shower start point
        '''
        self.output['shower_fragments']   = []
        self.output['leftover_energy']    = []
        self.output['leftover_fragments'] = []
        if not len(self.output['shower_mask']):
            return
        # Assign clusters
        points = self.output['energy']
        shower_starts = np.array([s.start for s in self.output['showers']])
        shower_points = self.output['energy'][self.output['shower_mask']]

        if self.cfg['shower_fragment'] == 'label':
            if not self.cfg['shower_start'] == 'label':
                raise ValueError('shower_fragment being "label" requires shower_start to be also "label"!')

            clusts = []
            showers = []
            for shower in self.output['showers']:
                pid = shower.pid
                # obtain the list of true cluster points
                mask = np.where(self.output['cluster_label'][:,self.IDX_CLUSTER_ID] == pid)[0]
                cluster = self.output['cluster_label'][mask]
                # now select shower energy depositions that is in the list (can't just slice by "mask" as size is different)
                cluster = self.select_overlap(shower_points,cluster)
                if len(cluster) < 1:
                    continue
                clusts.append(cluster)
                showers.append(shower)
            self.output['showers'] = showers
            self.output['shower_fragments'] = clusts

            # compute remaining points
            remain = shower_points
            if len(clusts):
                used = np.unique(np.concatenate(clusts)).astype(np.int32)
                remain = [i for i in range(len(shower_points)) if not i in used]
                remain = shower_points[remain]
            if len(remain) < 1:
                return
            remain_labels = self.frag_est.make_shower_frags(remain)
            remain_points = remain[np.where(remain_labels == -1)]
            self.output['leftover_energy'] = self.select_overlap(shower_points,remain_points)
            for idx in range(np.max(remain_labels)):
                remain_cluster = remain[np.where(remain_labels == idx)]
                self.output['leftover_fragments'].append(self.select_overlap(shower_points,remain_cluster))

        elif self.cfg['shower_fragment'] == 'dbscan':
            showers = []
            if len(shower_points)<1:
                self.output['showers'] = showers
            else:
                clusts, remaining_clusts, remaining_energy = self.frag_est.create_clusters(shower_points, shower_starts)
                assert len(clusts) == len(self.output['showers'])
                for idx, cluster in enumerate(clusts):
                    if len(cluster) < 1: continue
                    showers.append(self.output['showers'][idx])
                    self.output['shower_fragments'].append(cluster)
                self.output['showers'] = showers
                self.output['leftover_fragments'] = remaining_clusts
                self.output['leftover_energy']    = remaining_energy

        elif self.cfg['shower_fragment'] == 'gnn':
            mapping = {idx:i for (i, idx) in enumerate(self.output['shower_mask'][0])}
            clusts = np.array([np.array([mapping[i] for i in c]) for c in self.output['forward']['shower_fragments'][0]])
            from scipy.special import softmax
            node_scores = softmax(self.output['forward']['node_pred'][0], axis=1)
            primary_labels = np.zeros(len(node_scores), dtype=bool)
            group_ids = self.output['forward']['group_pred'][0]
            for g in np.unique(group_ids):
                mask = np.where(group_ids == g)[0]
                idx  = node_scores[mask][:,1].argmax()
                primary_labels[mask[idx]] = True
            primaries = np.where(primary_labels)[0]
            others = [i for i in range(len(clusts)) if i not in primaries]
            labels = -np.ones(len(shower_points))
            for i, c in enumerate(clusts):
                labels[c] = i
            self.output['shower_fragments'] = clusts[primaries]
            self.output['leftover_fragments'] = clusts[others]
            self.output['remaining_energy'] = np.where(labels == -1)[0]

        else:
            raise ValueError('Shower fragment reconstruction method not recognized:', self.cfg['shower_fragment'])


    def reconstruct_shower_directions(self, event):
        '''
        Reconstructs the direction of the showers
        '''
        if self.cfg['shower_direction'] == 'label':
            for shower in self.output['showers']:
                part = event['particles'][0][int(shower.pid)]
                mom = [part.px(), part.py(), part.pz()]
                shower.direction = list(np.array(mom)/np.linalg.norm(mom))

        elif self.cfg['shower_direction'] == 'pca' or self.cfg['shower_direction'] == 'cent':
            # Apply DBSCAN, PCA on the touching cluster to get angles
            algo = self.cfg['shower_direction']
            shower_points = self.output['energy'][self.output['shower_mask']]
            starts = np.array([s.start for s in self.output['showers']])
            fragments = [shower_points[inds] for inds in self.output['shower_fragments']]

            # Old method
            #try:
            #    res = self.dir_est.get_directions(starts, fragments, max_distance=float(10), mode=algo) #max_distance=float('inf')
            #except AssertionError as err: # Cluster was not found for at least one primary
            #    if self.verbose:
            #        print('Error in direction reconstruction:', err)
            #    res = [[0., 0., 0.] for _ in range(len(self.output['showers']))]

            #for i, shower in enumerate(self.output['showers']):
            #    shower.direction = res[i]
            
            # New method
            for i, shower in enumerate(self.output['showers']):
                direction = cluster_direction(torch.tensor(fragments[i]), torch.tensor(starts[i]), max_dist=-1, optimize=True)
                shower.direction = np.array(direction)

        else:
            raise ValueError('Shower direction reconstruction method not recognized:', self.cfg['shower_direction'])


    def reconstruct_shower_cluster(self,event):
        '''
        Cluster shower fragments and left-over pixels
        '''
        if len(self.output['shower_fragments']) < 1:
            return
        if self.cfg['shower_cluster'] == 'label':
            # Require the shower definition (= list of start points) match with true cluster definition
            if not self.cfg['shower_start'] == 'label':
                raise ValueError('shower_cluster value "label" must be combined with shower_start "label"!')
            # Obtain a true cluster
            segment = self.output['segment']
            shower_points = self.output['energy'][self.output['shower_mask']]

            particles = event['particles'][0]
            for shower in self.output['showers']:
                gid = particles[shower.pid].group_id()
                mask = np.where(self.output['cluster_label'][:,self.IDX_GROUP_ID] == gid)[0]
                points = self.output['cluster_label'][mask]
                valid_idx = self.select_overlap(shower_points,points)
                shower.voxels = valid_idx

        elif self.cfg['shower_cluster'] == 'cone':
            self.merge_fragments(event)
            self.merge_leftovers(event)

        elif self.cfg['shower_cluster'] == 'gnn':
            mapping = {idx:i for (i, idx) in enumerate(self.output['shower_mask'][0])}
            clusts = np.array([np.array([mapping[i] for i in c]) for c in self.output['forward']['shower_fragments'][0]])
            group_ids = self.output['forward']['group_pred'][0]
            frags, left_frags = [], []
            indices = []
            used_group_ids = []
            for i, sh in enumerate(self.output['showers']):
                for gr_id in group_ids:
                    if (gr_id == sh.group_pred) and (gr_id not in used_group_ids):
                        used_group_ids.append(gr_id)
                        indices.append(np.where(group_ids == sh.group_pred)[0])
            for frag, ind in enumerate(indices):
                frags.append(np.concatenate([clusts[j] for j in ind]))
            for i, s in enumerate(self.output['showers']):
                s.voxels = frags[i]
                #s.energy = np.sum(self.output['energy'][self.output['shower_mask']][frags[i]][:,-1])
            self.output['shower_fragments'] = frags
            self.output['leftover_fragments'] = left_frags
            
        else:
            raise ValueError('Merge shower fragments method not recognized:', self.cfg['shower_cluster'])


    def merge_fragments(self, event):
        '''
        Merge shower fragments with assigned start point
        '''
        from pi0.cluster.fragment_merger import group_fragments
        impact_parameter = float(self.cfg['shower_cluster_params']['IP'])
        radiation_length = float(self.cfg['shower_cluster_params']['Distance'])
        shower_points = self.output['energy'][self.output['shower_mask']]
        fragments = []
        for i, s in enumerate(self.output['showers']):
            start = s.start
            voxel = shower_points[self.output['shower_fragments'][i]]
            fragments.append([start,voxel])

        roots, groups, pairs = group_fragments(fragments, dist_prep=impact_parameter, dist_rad=radiation_length)
        assert(len(roots) == len(groups))
        # loop over groups and merge fragments
        showers = []
        fragments = []
        for idx,root in enumerate(roots):
            # merge secondaries
            showers.append(self.output['showers'][root])
            secondaries = [self.output['shower_fragments'][fidx] for fidx in groups[idx]]
            fragments.append(np.concatenate(secondaries))
        self.output['showers'] = showers
        self.output['shower_fragments'] = fragments


    def merge_leftovers(self, event):
        '''
        Merge leftover fragments (w/o start point) and leftover pixels
        '''
        # Fits cones to each shower, adds energies within that cone
        starts = np.array([s.start for s in self.output['showers']])
        dirs = np.array([s.direction for s in self.output['showers']])
        shower_energy = self.output['energy'][self.output['shower_mask']]
        #print(self.output['leftover_fragments'][0].type)
        remaining_inds = np.concatenate(self.output['leftover_fragments'] + [self.output['leftover_energy']]).astype(np.int32)
        if len(remaining_inds) < 1:
            for i, shower in enumerate(self.output['showers']):
                shower.voxels = self.output['shower_fragments'][i]
            return

        remaining_energy = shower_energy[remaining_inds]
        fragments = [shower_energy[ind] for ind in self.output['shower_fragments']]
        pred = self.clusterer.fit_predict(remaining_energy, starts, fragments, dirs)

        for i, shower in enumerate(self.output['showers']):
            merging_inds = remaining_inds[np.where(pred == i)]
            shower.voxels = np.concatenate([self.output['shower_fragments'][i],merging_inds])


    def reconstruct_shower_energy(self, event):

        if self.cfg['shower_energy'] == 'label':
            if not self.cfg['shower_start'] == 'label':
                raise ValueError('shower_energy value "label" must be combined with shower_start "label"!')
            particles = event['particles'][0]
            for s in self.output['showers']:
                s.energy = particles[s.pid].energy_init()

        elif self.cfg['shower_energy'] == 'pixel_sum':
            for s in self.output['showers']:
                s.energy = np.sum(self.output['energy'][self.output['shower_mask']][s.voxels][:,-1])

        else:
            raise ValueError('shower_energy method not recognized:', self.cfg['shower_energy'])


    def identify_pi0(self, event):       
        '''
        Proposes pi0 candidates (match two showers)
        '''
        self.output['matches']  = []
        self.output['vertices'] = []
        n_showers = len(self.output['showers'])
        if self.cfg['shower_match'] == 'label':
            # Make the pairs based on parent track id
            shower_lists = {}
            for idx, shower in enumerate(self.output['showers']):
                part = event['particles'][0][shower.pid]
                if not part.parent_pdg_code() == 111:
                    continue
                if not part.parent_track_id() in shower_lists:
                    shower_lists[part.parent_track_id()] = [part.position(),idx]
                else:
                    shower_lists[part.parent_track_id()].append(idx)

            for parent, pids in shower_lists.items():
                if len(pids) <= 2:
                    continue
                elif len(pids) == 3:
                    pair = pids[1:]
                    pos  = [pids[0].x(),pids[0].y(),pids[0].z()]
                    self.output['matches'].append(pids[1:])
                    self.output['vertices'].append(pos)
                if len(pids) > 3:
                    print('WARNING: in identify_pi0, ignoring >2 particle pairs from the shared parent...')
                    for p in pids[1:]:
                        print('ID =',p)
                        print(event['particles'][0][p].dump())

            """
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
            """
            
            return self.output['matches']

        elif self.cfg['shower_match'] == 'proximity':
            tolerance = 10. # Defines the maximum distance between a track-labeled edep and the reconstructed vertex
                            # TODO: Read this form chain.config?
                            # TODO: Optimise this parameter
            
            # Pair closest shower vectors
            sh_starts   = np.array([s.start for s in self.output['showers']])
            sh_dirs     = np.array([s.direction for s in self.output['showers']])
            sh_energies = np.array([s.energy for s in self.output['showers']])
            try:
                self.output['matches'], self.output['vertices'], dists = self.matcher.find_matches(sh_starts, sh_dirs, sh_energies, self.output['segment'], self.cfg['shower_match'], tolerance)
            except ValueError as err:
                if self.verbose:
                    print('Error in PID:', err)
                return

        elif self.cfg['shower_match'] == 'ppn':
            tolerance = 40. # Defines the maximum distance between the first principal component of pairs of showers at the closest point // a point from the PPN and a POCA
                            # TODO: Read this form chain.config?
                            # TODO: Optimise this parameter
            # Pair closest shower vectors
            sh_starts = np.array([s.start for s in self.output['showers']])
            sh_dirs = np.array([s.direction for s in self.output['showers']])
            sh_energies = np.array([s.energy for s in self.output['showers']])
            try:
                self.output['matches'], self.output['vertices'] = self.matcher.find_matches(sh_starts, sh_dirs, sh_energies, self.output['segment'], self.cfg['shower_match'], tolerance, self.output['PPN_track_points'])
            except ValueError as err:
                if self.verbose:
                    print('Error in PID:', err)
                return
            
        else:
            raise ValueError('Shower matching method not recognized:', self.cfg['shower_match'])

            
        if self.cfg['refit_dir'] and (self.cfg['shower_match'] == 'proximity' or self.cfg['shower_match'] == 'ppn'):
            for i, match in enumerate(self.output['matches']):
                idx1, idx2 = match
                v = np.array(self.output['vertices'][i])
                for shower_idx in [idx1,idx2]:
                    #new_dir = np.array(points[shower_idx]) - v
                    #new_dir = np.array(points[shower_idx][:3]) - v
                    new_dir = self.output['showers'][shower_idx].start - v
                    if np.all(new_dir==0):
                        if self.verbose:
                            print('INFO : ShowerStart == VertexPos -> Do not refit the direction of the shower ... (event:', self.event['index'], ')')
                        continue
                        
                    # only take new direction if shower start position is not too close to the vertex (makes it more accurate)
                    # TODO: Optimise this parameter
                    if np.linalg.norm(new_dir) < 5.:
                        continue
                    self.output['showers'][shower_idx].direction = new_dir/np.linalg.norm(new_dir)

            # Below commented out as the clustering stage relies on ordering of merging fragments and
            #       grouping of fragments. Simply re-calling that function at this point with an updated
            #       angle calculation may not be a good idea (i.e. could merge 2 big showers by repeating
            #       merge fragments inside the reconstruct_shower_cluster function).
            #if self.cfg['shower_energy'] == 'cone' and self.cfg['refit_cone']:
            #    self.reconstruct_shower_energy(event)
            
            
    def fiducialize(self, event):
        '''
        If a shower has edeps Out Of Fiducial Volume (OOFV), put the shower number to self.output['OOFV']
        '''
        self.output['OOFV'] = []
        
        if self.cfg['fiducialize'] > 0:
            #print(' Fiducialization: ', self.cfg['fiducialize'], ' pixels from boundary.')
            pass
        elif self.cfg['fiducialize'] == 0:
            #print(' No fiducialization to be done.')
            return self.output['OOFV']
        else:
            raise ValueError('fiducialize method (in chain.py) not recognized. Require integer >= 0. You entered:', self.cfg['fiducialize'])

        energy      = self.output['energy']
        shower_mask = self.output['shower_mask']

        # Obtain shower's info: x,y,z,batch_id,e_deposited
        shower_counter = 0
        for s in self.output['showers']: # s is shower object
            s.x        = energy[shower_mask][s.voxels][:,0]
            s.y        = energy[shower_mask][s.voxels][:,1]
            s.z        = energy[shower_mask][s.voxels][:,2]
            s.batch_id = energy[shower_mask][s.voxels][:,3]
            s.edep     = energy[shower_mask][s.voxels][:,4]
            coords     = np.array((s.x,s.y,s.z))

            # If at least one edep is OOFV: Put shower number to list self.output['OOFV']
            if ( np.any(coords<self.cfg['fiducialize']) or np.any(coords>(767-self.cfg['fiducialize'])) ):
                self.output['OOFV'].append(shower_counter)

            shower_counter += 1
        return self.output['OOFV']


    def pi0_mass(self):
        '''
        Reconstructs the pi0 mass
        '''
        from math import sqrt
        masses = []
        
        for match in self.output['matches']:
            idx1, idx2 = match

            # Do not use the pi0 decay if at least one of the showers has edeps OOFV:
            #if (idx1 in self.output['OOFV'] or idx2 in self.output['OOFV']):
            #    if self.verbose:
            #        print('Shower edeps close to LAr volume edge -> skip this pi0 in event ', self.event['index'])
            #    continue
            s1, s2 = self.output['showers'][idx1], self.output['showers'][idx2]
            e1, e2 = s1.energy, s2.energy
            t1, t2 = s1.direction, s2.direction
            costheta = np.dot(t1, t2)
            if abs(costheta) > 1.:
                print(' WARNING: costheta = np.dot(sh1.dir, sh2.dir) > 1. sh1.dir = ', t1, ', sh2.dir = ', t2, 'costheta = ', costheta)
                masses.append(-9)
                continue
            #if e1 < 35. or e2 < 35.:
            #    masses.append(-3)
            #    continue
            masses.append(sqrt(2.*e1*e2*(1.-costheta)))
        self.output['masses'] = masses
        return masses


    def extract_true_information(self, event):
        '''
        Obtain true informations about pi0s and gammas originated from pi0 decays and dump
        it to self.true_info['<variable>']
        '''
        import math
        import numpy as np

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
        if 2*self.true_info['n_pi0'] != self.true_info['n_gammas']:
            print(' WARNING: The number of pi0 (', self.true_info['n_pi0'], ') does not match the corresponding number of photons (', self.true_info['n_gammas'], ')... [event:', self.event['index'], ']')
        #assert 2*self.true_info['n_pi0'] == self.true_info['n_gammas'], " WARNING: The number of pi0 does not match the corresponding number of photons... "
        
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
        
        # Loop over all clusters and get voxels for every true gamma shower
        # Note: using parser 'parse_cluster3d_full', one can obtain a cluster via
        # clusters = self.event['cluster_label'] where the entries are
        # x,y,z,batch_id,voxel_value,cluster_id,group_id,semantic_type
        if self.true_info['n_gammas'] > 0:
            self.true_info['gamma_voxels'] = [[] for _ in range(self.true_info['n_gammas'])]
            # gamma_voxels is a list of n lists (n = number of gammas) with voxel coordinates of each shower
            clusters = self.event['cluster_label']
            for cluster_index, edep in enumerate(clusters):
                for group_index, group in enumerate(self.true_info['shower_group_ids']):
                    if edep[6] == group[0]:
                        self.true_info['gamma_voxels'][group_index].append([edep[0],edep[1],edep[2]])
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


    def extract_reco_information(self, event):
        '''
        Obtain reconstructed informations about pi0s and gammas originated from pi0 decays and dump
        it to self.reco_info['<variable>']
        '''
        import math

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
    
    def find_true_electron_showers(self, event):
        '''
        Obtain true information about showers induced by (primary) electrons and dump it to self.output['electronShowers'].
        '''
        electron_showers = []

        # Get group_id of shower produced by electron
        group_ids = []
        for i, p in enumerate(self.event['particles'][0]):
            if p.pdg_code() == 11 and p.parent_pdg_code() == 0:
                group_ids.append(p.group_id())
        
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
            if p.pdg_code() == 11 and p.parent_pdg_code() == 0:
                #print(' Electron: ')
                #print(p.dump())            
                _pid        = int(p.id())
                _group_id   = p.group_id()
                _start      = [p.x(), p.y(), p.z()]
                _mom_init   = [p.px(), p.py(), p.pz()]
                _dir        = [p.px(), p.py(), p.pz()]/np.linalg.norm([p.px(), p.py(), p.pz()])
                _voxels     = voxels[counter]
                _einit      = p.energy_init()
                _edeps      = edeps[counter]
                _edep_tot   = np.sum(edeps[counter])
                # TODO: check that shower starts within FV, add attribute if needed
                # TODO: check that only edeps within LAr volume are counted
                electron_showers.append(ElectronShower(pid=_pid,group_id=_group_id,start=_start,mom_init=_mom_init,direction=_dir,voxels=_voxels,einit=_einit,edeps=_edeps,edep_tot=_edep_tot))
                counter += 1
        self.output['electronShowers'] = electron_showers
        return
        
    def find_true_photon_showers(self, event):
        '''
        Obtain true information about showers induced by photons from a pi0 decay and dump it to self.output['photonShowers'].
        '''
        photon_showers = []
        
        # First, produce list of n lists (n=number of photons from pi0 decay) with ids and track_ids
        ids_of_true_photons  = []
        tids_of_true_photons = []
        group_ids            = []
        for i, p in enumerate(self.event['particles'][0]):
            if p.parent_pdg_code() == 111 and p.pdg_code() == 22:
                ids_of_true_photons.append(p.id())
                tids_of_true_photons.append(p.track_id())
                group_ids.append(p.group_id())
        
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
        #print(' ids_of_particles_in_shower:  ', ids_of_particles_in_shower)
        #print(' tids_of_particles_in_shower: ', tids_of_particles_in_shower)

        # Obtain shower's first (in time) energy deposit and define it as true shower's start position
        showers_start = []
        for j, IDs in enumerate(ids_of_particles_in_shower):
            min_time = float('inf')
            first_step = [float('inf'), float('inf'), float('inf')]
            for i, p in enumerate(self.event['particles'][0]):
                if p.id() in IDs:
                    if p.first_step().t() > 0. and p.first_step().t() < min_time:
                        min_time = p.first_step().t()
                        first_step = [p.first_step().x(), p.first_step().y(), p.first_step().z()] # , p.first_step().t()
            showers_start.append(first_step)
        #print(' showers_start:               ', showers_start)

        # Assign the photon induced shower's properties
        counter = 0
        for i, ID in enumerate(ids_of_true_photons):
            for j, p in enumerate(self.event['particles'][0]):
                if p.id() == ID:
                    _pid        = int(p.id())
                    _group_id   = p.group_id()
                    _start      = showers_start[i]
                    _mom_init   = [p.px(), p.py(), p.pz()]
                    _dir        = [p.px(), p.py(), p.pz()]/np.linalg.norm([p.px(), p.py(), p.pz()])
                    _voxels     = voxels[counter] # TODO: implement
                    _einit      = p.energy_init()
                    _edeps      = edeps[counter]
                    _edep_tot   = np.sum(edeps[counter])
                    # TODO: check that shower starts within FV, add attribute if needed
                    # TODO: check that only edeps within LAr volume are counted
                    photon_showers.append(PhotonShower(pid=_pid,group_id=_group_id,start=_start,mom_init=_mom_init,direction=_dir,voxels=_voxels,einit=_einit,edeps=_edeps,edep_tot=_edep_tot))
                    counter += 1
                    break
        self.output['photonShowers'] = photon_showers
        return


    def draw(self,**kargs):
        import plotly
        import numpy
        from mlreco.visualization.points import scatter_points
        import plotly.graph_objs as go
        from plotly.offline import iplot

        graph_data = []
        
        # Draw voxels with cluster labels
        energy = self.output['energy']
        shower_mask = self.output['shower_mask']
        graph_data += scatter_points(energy,markersize=2,color=energy[:,-1],colorscale='Inferno')
        graph_data[-1].name='Energy'
        
        # Add true pi0 decay points
        if len(self.true_info['gamma_pos'])>0:
            true_pi0_decays = self.true_info['gamma_pos']
            graph_data += scatter_points(numpy.asarray(true_pi0_decays),markersize=5, color='green')
            graph_data[-1].name = 'True pi0 decay vertices'
            
        # Add reconstructed pi0 decay points
        try:
            reco_pi0_decays = self.output['vertices']
            graph_data += scatter_points(numpy.asarray(reco_pi0_decays),markersize=7, color='lightgreen')
            graph_data[-1].name = 'Reconstructed pi0 decay vertices'
        except:
            pass
        
        '''
        # Add true photons 1st steps
        if len(self.true_info['gamma_first_step'])>0:
            true_gammas_first_steps = self.true_info['gamma_first_step']
            #print(' true_gammas_first_steps: ', true_gammas_first_steps)
            graph_data += scatter_points(numpy.asarray(true_gammas_first_steps), markersize=5, color='magenta')
            graph_data[-1].name = 'True photons 1st steps'
        
        # Add compton electrons 1st steps
        if len(self.true_info['compton_electron_first_step'])>0:
            compton_electrons_first_steps = self.true_info['compton_electron_first_step']
            #print(' compton_electrons_first_steps: ', compton_electrons_first_steps)
            graph_data += scatter_points(numpy.asarray(compton_electrons_first_steps), markersize=5, color='green')
            graph_data[-1].name = 'True compton electrons 1st steps'
        '''    
        # Add shower's true 1st (in time) step
        if len(self.true_info['shower_first_edep'])>0:
            shower_first_edep = self.true_info['shower_first_edep']
            #print(' shower_first_edep: ', shower_first_edep)
            graph_data += scatter_points(numpy.asarray(shower_first_edep), markersize=5, color='red')
            graph_data[-1].name = 'True showers 1st steps'
            
        # Add points predicted by PPN
        if self.output['PPN_track_points']:
            points = np.array([i.ppns for i in self.output['PPN_track_points']])
            graph_data += scatter_points(points,markersize=4, color='purple')
            graph_data[-1].name = 'PPN track points'
        '''
        if self.output['PPN_shower_points']:
            points = np.array([i.ppns for i in self.output['PPN_shower_points']])
            graph_data += scatter_points(points,markersize=4, color='purple')
            graph_data[-1].name = 'PPN shower points'
        if self.output['PPN_michel_points']:
            points = np.array([i.ppns for i in self.output['PPN_michel_points']])
            graph_data += scatter_points(points,markersize=4, color='purple')
            graph_data[-1].name = 'PPN michel points'
        if self.output['PPN_delta_points']:
            points = np.array([i.ppns for i in self.output['PPN_delta_points']])
            graph_data += scatter_points(points,markersize=4, color='purple')
            graph_data[-1].name = 'PPN delta points'
        if self.output['PPN_LEScat_points']:
            points = np.array([i.ppns for i in self.output['PPN_LEScat_points']])
            graph_data += scatter_points(points,markersize=4, color='purple')
            graph_data[-1].name = 'PPN LEScatter points'
        '''

        '''
        # Add true photon's directions (based on true pi0 decay vertex and true photon's 1st steps)
        if 'gamma_pos' in self.true_info and 'gamma_first_step' in self.true_info:
            for i, true_dir in enumerate(self.true_info['gamma_pos']):
                vertex = self.true_info['gamma_pos'][i]
                first_steps = self.true_info['gamma_first_step'][i]
                points = [vertex, first_steps]
                graph_data += scatter_points(np.array(points),markersize=4,color='blue')
                graph_data[-1].name = 'True photon %i: vertex to first step' % i
                graph_data[-1].mode = 'lines,markers'
        '''
        
        # Add true photon's directions (based on true pi0 decay vertex and true photon's 1st (in time) edep)
        if 'gamma_pos' in self.true_info and 'shower_first_edep' in self.true_info:
            for i, true_dir in enumerate(self.true_info['gamma_pos']):
                vertex = self.true_info['gamma_pos'][i]
                first_edeps = self.true_info['shower_first_edep'][i]
                points = [vertex, first_edeps]
                graph_data += scatter_points(np.array(points),markersize=4,color='green')
                graph_data[-1].name = 'True photon %i: vertex to first edep' % i
                graph_data[-1].mode = 'lines,markers'

        colors = plotly.colors.qualitative.Light24
        for i, s in enumerate(self.output['showers']):
            points = energy[shower_mask][s.voxels]
            color = colors[i % (len(colors))]
            graph_data += scatter_points(points,markersize=2,color=color)
            #graph_data[-1].name = 'Shower %d (id=%d, edep=%.2f)' % (i,s.pid,s.energy)
            graph_data[-1].name = 'Shower %d (id=%d, edep=%.2f, dir=[%.2f,%.2f,%.2f])' % (i,s.pid,s.energy,s.direction[0],s.direction[1],s.direction[2])

        if len(self.output['showers']):

            # Add EM primary points
            points = np.array([s.start for s in self.output['showers']])
            graph_data += scatter_points(points)
            graph_data[-1].name = 'Reconstructed shower starts'

            # Add EM primary directions
            dirs = np.array([s.direction for s in self.output['showers']])
            cone_start = points[:,:3]
            arrows = go.Cone(x=cone_start[:,0], y=cone_start[:,1], z=cone_start[:,2],
                             u=-dirs[:,0], v=-dirs[:,1], w=-dirs[:,2],
                             sizemode='absolute', sizeref=1.0, anchor='tip',
                             showscale=False, opacity=0.4)
            graph_data.append(arrows)

            # Add a vertex if matches, join vertex to start points
            if 'matches' in self.output:
                for i, match in enumerate(self.output['matches']):
                    v = self.output['vertices'][i]
                    idx1, idx2 = match
                    s1, s2 = self.output['showers'][idx1].start, self.output['showers'][idx2].start
                    points = [v, s1, v, s2]
                    graph_data += scatter_points(np.array(points),color='red')
                    graph_data[-1].name = 'Reconstructed pi0 (%.2f MeV)' % self.output['masses'][i]
                    graph_data[-1].mode = 'lines,markers'

        # Draw
        iplot(go.Figure(data=graph_data,layout=self.layout(**kargs)))


#    @staticmethod
#    def is_shower(particle):
#        '''
#        Check if the particle is a shower
#        '''
#        pdg_code = abs(particle.pdg_code())
#        if not pdg_code == 22 and not pdg_code == 11 :
#            return False
#
#        return False