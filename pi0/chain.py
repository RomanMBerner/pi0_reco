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
#from .identification.matcher_old import Pi0Matcher # Pi0 vertex is chosen as the PPN point closest to the CPA of two showers
from .identification.matcher import Pi0Matcher # Pi0 vertex is chosen as the PPN point which is in 'best' angular agreement with a pair of showers
from .identification.PID import ElectronPhoton_Separation
from .analyse.analyser import * #Analyser
from mlreco.main_funcs import process_config, prepare, apply_event_filter
from mlreco.utils import CSVData
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.gnn.cluster import cluster_direction



# Chain object class that loads and stores the chain parameters
class Pi0Chain():

    #Class constants
    IDX_SEMANTIC_ID = -1
    IDX_CLUSTER_ID = 6 # Old: IDX_GROUP_ID    = -2
    IDX_CLUSTER_ID = 5 # Old: IDX_CLUSTER_ID  = -3
    

    def __init__(self, io_cfg, chain_cfg, verbose=False):
        '''
        Initializes the chain from the configuration file
        '''
        # Initialize the data loader
        io_cfg = yaml.load(io_cfg,Loader=yaml.Loader)

        # Save config, initialize output
        self.cfg       = chain_cfg
        self.verbose   = verbose
        self.event     = None
        self.output    = {}
        self.true_info = {}
        self.reco_info = {}

        # Initialize log
        # TODO: Check if this is needed any longer
        log_path = chain_cfg['name']+'_log.csv'
        log_path = 'masses_fiducialized_' + str(chain_cfg['fiducialize']) + 'px.csv'
        print('Initialized Pi0 mass chain, log path:', log_path)
        self._log = CSVData(log_path)
        self._keys = ['event_id', 'pion_id', 'pion_mass']

        # If a network is specified, initialize the network
        self.network = False
        
        if chain_cfg['modules']['segment']['method'] == 'uresnet' or chain_cfg['modules']['shower_start']['method'] == 'ppn':
            self.network = True
            with open(chain_cfg['mlreco']['cfg_path']) as cfg_file:
                net_cfg = yaml.load(cfg_file,Loader=yaml.Loader)
            io_cfg['model'] = net_cfg['model']
            io_cfg['trainval'] = net_cfg['trainval']

        # Initialize the fragment identifier
        self.frag_est = DBSCANCluster()

        # If a direction estimator is requested, initialize it
        if chain_cfg['modules']['shower_direction']['method'] != 'label':
            self.dir_est = DirectionEstimator()

        # If a clusterer is requested, initialize it
        if chain_cfg['modules']['shower_cluster']['method'] in ['cone', 'gnn']:
            self.clusterer = ConeClusterer()

        # If a pi0 identifier is requested, initialize it
        if (chain_cfg['modules']['shower_match']['method'] == 'proximity' or chain_cfg['modules']['shower_match']['method'] == 'ppn'):
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
        
        # Check input
        self.infer_inputs(event)

        # Set the semantics
        self.infer_semantics(event)

        # Analyser module for true qantities
        if (self.cfg['analyse_true']):
            self.analyser = Analyser()
            Analyser.initialize_true(self)
            Analyser.extract_true_information(self, event)
            Analyser.find_true_electron_showers(self, event)
            Analyser.find_true_photon_showers(self, event)
            
        # Initialise also analyser module for reconstructed quantities
        if (self.cfg['analyse_reco']):
            Analyser.initialize_reco(self)
        
        # Filter out ghosts
        # TODO: This function can be removed from the pixel branch
        self.filter_ghosts(event)
        
        # Obtain points from PPN
        if (self.cfg['modules']['shower_start']['method'] == 'ppn' or self.cfg['modules']['shower_match']['method'] == 'ppn'):
            self.obtain_ppn_points(event)

        # Reconstruct energy
        self.charge_to_energy(event)

        # Check data dimensions
        assert self.output['energy' ].shape == self.output['charge'].shape
        assert self.output['energy' ].shape == self.output['segment'].shape

        # Identify cluster start points, skip if there are less than 2 (no pi0)
        self.reconstruct_cluster_starts(event)
        if len(self.output['showers']) < 1: # TODO: 1 only for testing purposes; change to 2
            if self.verbose:
                print('< 1 shower start points found in event', event_id)
            return

        # Form shower fragments
        self.reconstruct_shower_fragments(event)
        if len(self.output['showers']) < 1:
            if self.verbose:
                print('< 1 shower fragment found in event', event_id)
            return
        
        # Reconstruct shower direction vectors
        self.reconstruct_shower_directions(event)

        # Reconstruct shower cluster
        self.reconstruct_shower_cluster(event)

        # Reconstruct shower energy
        self.reconstruct_shower_energy(event)
        
        # Identify shower start points, skip if there are less than 2 (no pi0)
        self.reconstruct_shower_starts(event)
        if len(self.output['showers']) < 1: # TODO: 1 only for testing purposes; change to 2
            if self.verbose:
                print('< 1 shower start points found in event', event_id)
            return
        
        # Reconstruct shower likelihood fractions (electron/positron like or photon like)
        self.obtain_likelihood(event)

        # Identify pi0 decays
        #print(' ------------------------------ ')
        #print(' Event ID: ', event_id)
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
        
        # Analyser module for reconstructed quantities
        if (self.cfg['analyse_reco']):
            Analyser.extract_reco_information(self, event)
        

    def infer_inputs(self,event):
        if self.cfg['modules']['deghost']['method'] == "label" or self.cfg['modules']['segment']['method'] == 'label':
            assert 'segment_label' in event
        if self.cfg['modules']['charge2e']['method'] == 'label':
            assert 'energy_label' in event
            self.output['energy_label'] = copy(event['energy_label'])
        if 'label' in [ self.cfg['modules']['shower_fragment']['method'], self.cfg['modules']['shower_direction']['method'],
                        self.cfg['modules']['shower_cluster']['method'], self.cfg['modules']['shower_energy']['method'] ]:
            assert self.cfg['modules']['shower_start']['method'] == 'label'
        if self.cfg['modules']['shower_start']['method'] == 'label':
            assert 'particles' in event
        if self.cfg['modules']['shower_fragment']['method'] == 'label' or self.cfg['modules']['shower_cluster']['method'] == 'label':
            assert 'cluster_label' in event
            self.output['cluster_label'] = copy(event['cluster_label'])

        self.output['charge'] = copy(event['input_data'])

        assert not 'segment_label' in event or event['segment_label'].shape == event['input_data'].shape
        assert not 'energy_label'  in event or event['energy_label' ].shape == event['input_data'].shape


    def infer_semantics(self, event):
        if self.cfg['modules']['segment']['method'] == 'label':
            self.output['segment'] = event['segment_label']

        elif self.cfg['modules']['segment']['method'] == 'uresnet':
            # Get the segmentation output of the network
            res = self.output['forward']
            # Argmax to determine most probable label
            self.output['segment'] = copy(event['segment_label'])
            self.output['segment'][:,-1] = np.argmax(res['segmentation'][0], axis=1)
        else:
            raise ValueError('Semantic segmentation method not recognized:', self.cfg['modules']['segment']['method'])

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
        if self.cfg['modules']['deghost']['method'] == 'label':
            mask = np.where(event['segment_label'][:,-1] != 5)[0]

        elif self.cfg['modules']['deghost']['method'] == 'uresnet':
            # Get the segmentation output of the network
            res = self.output['forward']
            # Argmax to determine most probable label
            pred_ghost = np.argmax(res['ghost'][0], axis=1)
            mask = np.where(pred_ghost == 0)[0]
        
        elif self.cfg['modules']['deghost']['method'] == 'none':
            # no de-ghosting needed: return!
            return
        
        else:
            raise ValueError('De-ghosting method not recognized:', self.cfg['modules']['deghost']['method'])

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
        shower_score_index = -1 * (int(larcv.kShapeUnknown) - int(larcv.kShapeShower))
        track_score_index  = -1 * (int(larcv.kShapeUnknown) - int(larcv.kShapeTrack))
        michel_score_index = -1 * (int(larcv.kShapeUnknown) - int(larcv.kShapeMichel))
        delta_score_index  = -1 * (int(larcv.kShapeUnknown) - int(larcv.kShapeDelta))
        LEScat_score_index = -1 * (int(larcv.kShapeUnknown) - int(larcv.kShapeLEScatter))
            
        points = uresnet_ppn_type_point_selector([event['input_data']][0],self.output['forward'])
        
        default_scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        
        try:
            thresholds = self.cfg['PPN_score_thresh']
            if not (len(thresholds)==5):
                print(' Need exactly 5 thresholds for PPN scores,', len(thresholds), 'given. Set them to', default_scores, ' ... ')
                thresholds = default_scores
        except:
            print(' Thresholds for PPN scores not defined in chain cfg. Set them to', default_scores, ' ... ')
            thresholds = default_scores

        shower_points = points[np.where(points[:,shower_score_index] > float(thresholds[0]))]
        track_points  = points[np.where(points[:,track_score_index]  > float(thresholds[1]))]
        michel_points = points[np.where(points[:,michel_score_index] > float(thresholds[2]))]
        delta_points  = points[np.where(points[:,delta_score_index]  > float(thresholds[3]))]
        LEScat_points = points[np.where(points[:,LEScat_score_index] > float(thresholds[4]))]
            
        shower_total_score = shower_points[:,shower_score_index] # * shower_points[:,point_score_index]
        track_total_score  = track_points[:,track_score_index]   # * track_points[:,point_score_index]
        michel_total_score = michel_points[:,michel_score_index] # * michel_points[:,point_score_index]
        delta_total_score  = delta_points[:,delta_score_index]   # * delta_points[:,point_score_index]
        LEScat_total_score = LEScat_points[:,LEScat_score_index] # * LEScat_points[:,point_score_index]
            
        shower_ordered = np.argsort(shower_total_score)
        track_ordered  = np.argsort(track_total_score)
        michel_ordered = np.argsort(michel_total_score)
        delta_ordered  = np.argsort(delta_total_score)
        LEScat_ordered = np.argsort(LEScat_total_score)
            
        self.output['PPN_shower_points'] = [ShowerPoints(ppns=shower_points[i,:3],shower_score=shower_total_score[i],shower_id=int(i)) for i in shower_ordered]
        self.output['PPN_track_points']  = [TrackPoints(ppns=track_points[i,:3],track_score=track_total_score[i],track_id=int(i)) for i in track_ordered]
        self.output['PPN_michel_points'] = [MichelPoints(ppns=michel_points[i,:3],michel_score=michel_total_score[i],michel_id=int(i)) for i in michel_ordered]
        self.output['PPN_delta_points']  = [DeltaPoints(ppns=delta_points[i,:3],delta_score=delta_total_score[i],delta_id=int(i)) for i in delta_ordered]
        self.output['PPN_LEScat_points'] = [LEScatPoints(ppns=LEScat_points[i,:3],LEScat_score=LEScat_total_score[i],LEScat_id=int(i)) for i in LEScat_ordered]


    def charge_to_energy(self, event):
        '''
        Reconstructs energy deposition from charge
        '''
        if self.cfg['modules']['charge2e']['method'] == 'none':
            self.output['energy'] = copy(self.output['charge'])

        elif self.cfg['modules']['charge2e']['method'] == 'label':
            self.output['energy'] = self.output['energy_label']

        elif self.cfg['modules']['charge2e']['method'] == 'constant':
            reco = self.cfg['modules']['charge2e']['cst']*self.output['charge'][:,-1]
            self.output['energy'] = copy(self.output['charge'])
            self.output['energy'][:,-1] = reco

        elif self.cfg['modules']['charge2e']['method'] == 'average':
            self.output['energy'] = copy(self.output['charge'])
            self.output['energy'][:,-1] = self.cfg['modules']['charge2e']['average']

        elif self.cfg['modules']['charge2e']['method'] == 'full':
            raise NotImplementedError('Proper energy reconstruction not implemented yet')

        elif self.cfg['modules']['charge2e']['method'] == 'enet':
            raise NotImplementedError('ENet not implemented yet')

        else:
            raise ValueError('Energy reconstruction method not recognized:', self.cfg['modules']['charge2e']['method'])

        
    def reconstruct_cluster_starts(self, event):
        '''
        Identify starting points of showers. Points should be ordered by the definiteness of a shower
        '''        
        if self.cfg['modules']['shower_start']['method'] == 'label':
            # Find showers
            particles = event['particles'][0]
            points = event['ppn_label']
            points = points[np.where(points[:,-2]==larcv.kShapeShower)]
            order  = np.argsort([particles[int(points[i,-1])].energy_deposit() for i in range(len(points))])
            if not self.cfg['modules']['shower_cluster']['method'] == 'label':
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

        # Old method:
        #elif self.cfg['modules']['shower_start']['method'] == 'ppn':
        #    from mlreco.utils.ppn import uresnet_ppn_type_point_selector
        #    shower_score_index = -1 * (int(larcv.kShapeUnknown) - int(larcv.kShapeShower))
        #    point_score_index  = -1 * (int(larcv.kShapeUnknown) + 1)
        #    points = uresnet_ppn_type_point_selector([event['input_data']],self.output['forward'])
        #    #points = points[np.where(points[:,shower_score_index] > self.cfg.get('ppn_shower_score_thresh',0.5))]
        #    points = points[0][np.where(points[0][:,shower_score_index] > self.cfg.get('ppn_shower_score_thresh',0.5))]
        #    total_score = points[:,shower_score_index] * points[:,point_score_index]
        #    order  = np.argsort(total_score)
        #    self.output['showers'] = [Shower(start=points[i,:3],pid=int(i)) for i in order]
        
        # New method:
        elif self.cfg['modules']['shower_start']['method'] == 'ppn':
            # Getting start points from PPN
            start_points = []
            
            # Use the node predictions to find primary nodes
            if not 'shower_node_pred' in self.output['forward']:
                self.output['showers'] = []
                return
            from scipy.special import softmax
            #node_scores = softmax(self.output['forward']['shower_node_pred'][0], axis=1)
            node_scores = softmax(self.output['forward']['shower_node_pred'][0], axis=1)
            primary_labels = np.zeros(len(node_scores), dtype=bool)
            group_ids = self.output['forward']['shower_group_pred'][0]
            for g in np.unique(group_ids):
                mask = np.where(group_ids == g)[0]
                idx  = node_scores[mask][:,1].argmax()
                primary_labels[mask[idx]] = True
            primaries = np.where(primary_labels)[0]
            primary_clusts = self.output['forward']['shower_fragments'][0][primaries]
            all_clusts = np.array([np.array(c) for c in self.output['forward']['shower_fragments'][0]], dtype="object")
            #group_ids = self.output['forward']['shower_group_pred'][0]
            
            # Obtain group_id predicted by GNN for all primary_clusts
            primary_clusts_shower_group_pred = []
            for i, primary_cluster in enumerate(primary_clusts):
                for j, all_cluster in enumerate(all_clusts):
                    if all_cluster[0]==primary_cluster[0]:
                        primary_clusts_shower_group_pred.append(group_ids[j])
                        break
            
            for clust_index, clust in enumerate(primary_clusts):
                # Get the energy deposits of the cluster
                clust_energy_pos = self.output['energy'][:,:3][clust] + 0.5 # +0.5 in order to get to voxel middle

                # Get the scores of PPN
                scores = softmax(self.output['forward']['points'][0][clust,3:5], axis=1)
                argmax = (scores[:,-1]).argmax()
                pos = self.output['energy'][clust][argmax,:3] + self.output['forward']['points'][0][clust][argmax,:3] + 0.5 # +0.5 (middle of voxel)
                #print(' scores:         ', scores)
                #print(' argmax:         ', argmax)
                #print(' scores[argmax]: ', scores[argmax])
                #print(' pos:            ', pos)
                start_points.append(pos)
            self.output['showers'] = [Shower(start=p,pid=int(i),shower_group_pred=int(primary_clusts_shower_group_pred[i])) for i, p in enumerate(start_points)]
        
        #'''
        elif self.cfg['modules']['shower_start']['method'] == 'gnn':
            # Use the node predictions to find primary nodes
            if not 'shower_node_pred' in self.output['forward']:
                self.output['showers'] = []
                return
            from scipy.special import softmax
            node_scores = softmax(self.output['forward']['shower_node_pred'][0], axis=1)
            primary_labels = np.zeros(len(node_scores), dtype=bool)
            group_ids = self.output['forward']['shower_group_pred'][0]
            for g in np.unique(group_ids):
                mask = np.where(group_ids == g)[0]
                idx  = node_scores[mask][:,1].argmax()
                primary_labels[mask[idx]] = True
            primaries = np.where(primary_labels)[0]
            primary_clusts = self.output['forward']['shower_fragments'][0][primaries]
            all_clusts = np.array([np.array(c) for c in self.output['forward']['shower_fragments'][0]])
            group_ids = self.output['forward']['shower_group_pred'][0]
            
            # Obtain group_id predicted by GNN for all primary_clusts
            primary_clusts_shower_group_pred = []
            for i, primary_cluster in enumerate(primary_clusts):
                for j, all_cluster in enumerate(all_clusts):
                    if all_cluster[0]==primary_cluster[0]:
                        primary_clusts_shower_group_pred.append(group_ids[j])
                        break

            # Getting start points from GNN
            start_finder = StartPointFinder()
            start_points = start_finder.find_start_points(self.output['energy'][:,:3], primary_clusts)
            self.output['showers'] = [Shower(start=p,pid=int(i),shower_group_pred=int(primary_clusts_shower_group_pred[i])) for i, p in enumerate(start_points)]
        #'''
        
        else:
            raise ValueError('EM shower primary identifiation method not recognized:', self.cfg['modules']['shower_start']['method'])


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

        if self.cfg['modules']['shower_fragment']['method'] == 'label':
            if not self.cfg['modules']['shower_start']['method'] == 'label':
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

        elif self.cfg['modules']['shower_fragment']['method'] == 'dbscan':
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

        elif self.cfg['modules']['shower_fragment']['method'] == 'gnn':
            mapping = {idx:i for (i, idx) in enumerate(self.output['shower_mask'][0])}
            clusts = np.array([np.array([mapping[i] for i in c]) for c in self.output['forward']['shower_fragments'][0]], dtype="object")
            from scipy.special import softmax
            node_scores = softmax(self.output['forward']['shower_node_pred'][0], axis=1)
            primary_labels = np.zeros(len(node_scores), dtype=bool)
            group_ids = self.output['forward']['shower_group_pred'][0]
            for g in np.unique(group_ids):
                mask = np.where(group_ids == g)[0]
                idx  = node_scores[mask][:,1].argmax()
                primary_labels[mask[idx]] = True
            primaries = np.where(primary_labels)[0]
            others = [i for i in range(len(clusts)) if i not in primaries]
            labels = -np.ones(len(shower_points))
            #print(' shower_points: ', shower_points)
            #print(' clusts: ', clusts)
            for i, c in enumerate(clusts):
                for num_ind, num in enumerate(c):
                    labels[num] = i
            #print(' labels: ', labels)
            self.output['shower_fragments'] = clusts[primaries]
            self.output['leftover_fragments'] = clusts[others]
            self.output['remaining_energy'] = np.where(labels == -1)[0]

        else:
            raise ValueError('Shower fragment reconstruction method not recognized:', self.cfg['modules']['shower_fragment']['method'])


    def reconstruct_shower_directions(self, event):
        '''
        Reconstructs the direction of the showers
        '''
        if self.cfg['modules']['shower_direction']['method'] == 'label':
            for shower in self.output['showers']:
                part = event['particles'][0][int(shower.pid)]
                mom = [part.px(), part.py(), part.pz()]
                shower.direction = list(np.array(mom)/np.linalg.norm(mom))

        elif self.cfg['modules']['shower_direction']['method'] == 'geo':
            # Apply DBSCAN, PCA on the touching cluster to get angles
            algo = self.cfg['modules']['shower_direction']['method']
            shower_points = self.output['energy'][self.output['shower_mask']]
            starts = np.array([s.start for s in self.output['showers']])
            
            #fragments = [shower_points[inds] for inds in self.output['shower_fragments']]
            # TODO: above method is not working anymore... -> Check the reason!
            # (IndexError: arrays used as indices must be of integer (or boolean) type)
            fragments = []
            for ind, frag in enumerate(self.output['shower_fragments']):
                fragments.append([shower_points[i] for i in frag])
            


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
            raise ValueError('Shower direction reconstruction method not recognized:', self.cfg['modules']['shower_direction']['method'])


    def reconstruct_shower_cluster(self,event):
        '''
        Cluster shower fragments and left-over pixels
        '''
        if len(self.output['shower_fragments']) < 1:
            return
        if self.cfg['modules']['shower_cluster']['method'] == 'label':
            # Require the shower definition (= list of start points) match with true cluster definition
            if not self.cfg['modules']['shower_start']['method'] == 'label':
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

        elif self.cfg['modules']['shower_cluster']['method'] == 'cone':
            self.merge_fragments(event)
            self.merge_leftovers(event)

        elif self.cfg['modules']['shower_cluster']['method'] == 'gnn':
            mapping = {idx:i for (i, idx) in enumerate(self.output['shower_mask'][0])}
            clusts = np.array([np.array([mapping[i] for i in c]) for c in self.output['forward']['shower_fragments'][0]], dtype="object")
            group_ids = self.output['forward']['shower_group_pred'][0]
            frags, left_frags = [], []
            indices = []
            used_group_ids = []
            for i, sh in enumerate(self.output['showers']):
                for gr_id in group_ids:
                    if (gr_id == sh.shower_group_pred) and (gr_id not in used_group_ids):
                        used_group_ids.append(gr_id)
                        indices.append(np.where(group_ids == sh.shower_group_pred)[0])
            for frag, ind in enumerate(indices):
                frags.append(np.concatenate([clusts[j] for j in ind]))
            for i, s in enumerate(self.output['showers']):
                s.voxels = frags[i]
                #s.energy = np.sum(self.output['energy'][self.output['shower_mask']][frags[i]][:,-1])
            self.output['shower_fragments'] = frags
            self.output['leftover_fragments'] = left_frags
            
        else:
            raise ValueError('Merge shower fragments method not recognized:', self.cfg['modules']['shower_cluster']['method'])


    def merge_fragments(self, event):
        '''
        Merge shower fragments with assigned start point
        '''
        from pi0.cluster.fragment_merger import group_fragments
        impact_parameter = float(self.cfg['modules']['shower_cluster']['IP'])
        radiation_length = float(self.cfg['modules']['shower_cluster']['Distance'])
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

        if self.cfg['modules']['shower_energy']['method'] == 'label':
            if not self.cfg['modules']['shower_start']['method'] == 'label':
                raise ValueError('shower_energy value "label" must be combined with shower_start "label"!')
            particles = event['particles'][0]
            for s in self.output['showers']:
                s.energy = particles[s.pid].energy_init()

        elif self.cfg['modules']['shower_energy']['method'] == 'pixel_sum':
            for s in self.output['showers']:
                #s.energy = np.sum(self.output['energy'][self.output['shower_mask']][s.voxels][:,-1])
                # TODO: above method is not working anymore... -> Check the reason!
                _energy = 0.
                for ind in s.voxels:
                    _energy += (self.output['energy'][self.output['shower_mask']])[ind,-1]
                s.energy = _energy
                
        else:
            raise ValueError('shower_energy method not recognized:', self.cfg['modules']['shower_energy']['method'])
    
    
    def reconstruct_shower_starts(self, event):
        '''
        Identify starting points of showers.
        Observation: Shower start point often is not correctly reconstructed.
        This happens in particular when DBSCAN cannot distinguish single fragments but the PPN can.
        The shower start is reconstructed in the following way (if self.cfg['modules']['shower_start']['method'] == 'ppn'):
            - From the PPN network, take all track-like and shower-like points (with score > threshold defined in config string)
            - From this set of points define a set of start_point_candidates (the candidates need to have >10 own edeps closer than 10 px)
            - If > 1 candidate points are present: Select the one which is at the 'edge' of the edep cloud
              Procedure:
              Integrate all charge along the PC in positive and negative direction (from the start_point_candidate).
              Take as shower start the candidate for which the ratio of the two numbers (smaller/larger) is the smallest.
        '''
        if self.cfg['modules']['shower_start']['method'] == 'ppn':
            
            # Get track-labeled and shower-labeled points from PPN
            if self.output['PPN_track_points']:
                track_points = np.array([i.ppns for i in self.output['PPN_track_points']])
                #print(' track_points: ', track_points)
            if self.output['PPN_shower_points']:
                shower_points = np.array([i.ppns for i in self.output['PPN_shower_points']])
                #print(' shower_points: ', shower_points)
                
            # Loop over all showers and select those PPN points which have >10 edeps within a radius of 10 pixels:
            for sh_index, sh in enumerate(self.output['showers']):
                # Get the energy deposits of the shower
                #edep_pos = self.output['energy'][self.output['shower_mask']][sh.voxels][:,0:3] + 0.5
                # TODO: above method is not working anymore... -> Check the reason!
                edep_pos = np.array([self.output['energy'][self.output['shower_mask']][i,0:3]+0.5 for i in sh.voxels])

                # Get start point candidates
                start_point_candidates = []
                if len(self.output['PPN_track_points'])>0:
                    for index, point in enumerate(track_points):
                        edep_counter = 0
                        for edep_index, edep in enumerate(edep_pos):
                            if np.linalg.norm(np.array(point)-np.array(edep)) < 10.:
                                edep_counter += 1
                        if edep_counter > 10:
                            start_point_candidates.append(point)
                if len(self.output['PPN_shower_points'])>0:
                    for index, point in enumerate(shower_points):
                        edep_counter = 0
                        for edep_index, edep in enumerate(edep_pos):
                            if np.linalg.norm(np.array(point)-np.array(edep)) < 10.:
                                edep_counter += 1
                        if edep_counter > 10:
                            start_point_candidates.append(point)
                
                # If > 1 candidate is present: Select one candidate as start point
                if len(start_point_candidates) > 1:
                    edep_counter_min = float('inf') #
                    candidate_index = 0

                    # Principal components analysis of the shower edeps
                    from scipy import linalg as LA
                    # Get barycentre of the cluster and obtain the covariance matrix
                    barycentre = edep_pos.mean(axis=0)
                    edep_pos_barycentre = edep_pos - barycentre # edep positions w.r.t. the barycentre
                    cov_mat = np.cov(edep_pos_barycentre, rowvar=False)
                    # Obtain (sorted) eigenvalues and eigenvectors (use 'eigh' rather than 'eig' since cov_mat is symmetric, gain in performance)
                    evals, evecs = LA.eigh(cov_mat)
                    idx = np.argsort(evals)[::-1]
                    evals = evals[idx] # sorted eigenvalues
                    evecs = evecs[:,idx] # sorted eigenvectors
                    PC_0 = np.array([evecs[0][0],evecs[1][0],evecs[2][0]]) # First principal component
                    PC_1 = np.array([evecs[0][1],evecs[1][1],evecs[2][1]]) # Second principal component
                    PC_2 = np.array([evecs[0][2],evecs[1][2],evecs[2][2]]) # Third principal component
                    
                    # Loop over all start point candidates and sum up all hits along the principal component ('left' and 'right' of the start_point_candidate)
                    min_fraction = float('inf')
                    selected_start = sh.start
                    for cand_ind, cand_pos in enumerate(start_point_candidates):
                        sum_left = 0
                        sum_right = 0
                        # Loop over all edeps and see whether edep is on the 'left' or on the 'right' side (along the PC_0) of the candidate_
                        for edep_ind, edep_position in enumerate(edep_pos):
                            _edep_position = edep_position - cand_pos # edep position, seen from candidate_position
                            if np.dot(_edep_position, PC_0)/(np.linalg.norm(_edep_position)*np.linalg.norm(PC_0))>0:
                                sum_left += 1
                            if np.dot(_edep_position, PC_0)/(np.linalg.norm(_edep_position)*np.linalg.norm(PC_0))<0:
                                sum_right +=1
                        # Take the fraction smaller_sum / larger_sum.
                        if sum_left < sum_right and sum_right != 0:
                            fraction = sum_left / sum_right
                        elif sum_left >= sum_right and sum_left != 0:
                            fraction = sum_right / sum_left
                        else:
                            print(' WARNING: Either sum_right or sum_left == 0 ... ')
                        if fraction < min_fraction:
                            min_fraction = fraction
                            selected_start = cand_pos
                    sh.start = selected_start
                    
            # Note: The start positions of some showers might have changed -> Execute functions relevant to this change again:
            self.reconstruct_shower_fragments(event)
            self.reconstruct_shower_directions(event)
            
        else:
            return

            
    def obtain_likelihood(self, event):
        '''
        Obtain the electron/positron- and photon likelihood fractions for the showers by looking at the dE/dx value at the very start of each EM shower.
        '''
        #energy      = self.output['energy']
        #shower_mask   = self.output['shower_mask']
        shower_energy = self.output['energy'][self.output['shower_mask']]
        reco_showers = self.output['showers']
        
        PID = ElectronPhoton_Separation()
        PID.likelihood_fractions(reco_showers, shower_energy)
        
        return


    def identify_pi0(self, event):       
        '''
        Proposes pi0 candidates (match two showers)
        '''
        self.output['matches']  = []
        self.output['vertices'] = []
        n_showers = len(self.output['showers'])
        if self.cfg['modules']['shower_match']['method'] == 'label':
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

        elif self.cfg['modules']['shower_match']['method'] == 'proximity':
            # Pair closest shower vectors
            sh_starts   = np.array([s.start for s in self.output['showers']])
            sh_dirs     = np.array([s.direction for s in self.output['showers']])
            sh_energies = np.array([s.energy for s in self.output['showers']])
            try:
                self.output['matches'], self.output['vertices'], dists = self.matcher.find_matches(self.output['showers'],\
                                                                                                   self.output['segment'],\
                                                                                                   self.cfg['modules']['shower_match']['method'],\
                                                                                                   self.cfg['modules']['shower_match']['verbose'])
            except ValueError as err:
                if self.verbose:
                    print('Error in PID:', err)
                return

        elif self.cfg['modules']['shower_match']['method'] == 'ppn':
            # Pair closest shower vectors
            sh_starts   = np.array([s.start for s in self.output['showers']])
            sh_dirs     = np.array([s.direction for s in self.output['showers']])
            sh_energies = np.array([s.energy for s in self.output['showers']])
            '''
            print(' sh_starts: ')
            for sh_index, start in enumerate(sh_starts):
                print('     ', start)
            print(' sh_directions: ')
            for sh_index, direction in enumerate(sh_dirs):
                print('     ', direction)
            print(' sh_energies: ')
            for sh_index, energy in enumerate(sh_energies):
                print('     ', energy)
            '''
            self.output['matches'], self.output['vertices'] = self.matcher.find_matches(self.output['showers'],\
                                                                                        self.output['segment'],\
                                                                                        self.cfg['modules']['shower_match']['method'],\
                                                                                        self.cfg['modules']['shower_match']['verbose'],\
                                                                                        self.output['PPN_track_points'])
        else:
            raise ValueError('Shower matching method not recognized:', self.cfg['modules']['shower_match']['method'])

        if self.cfg['modules']['shower_match']['refit_dir'] and\
          (self.cfg['modules']['shower_match']['method'] == 'proximity' or self.cfg['modules']['shower_match']['method'] == 'ppn'):
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
                        
                    # Only take new direction if shower start position is not too close to the vertex
                    # TODO: Optimise this parameter
                    if np.linalg.norm(new_dir) < 5.:
                        continue
                    self.output['showers'][shower_idx].direction = new_dir/np.linalg.norm(new_dir)

            # Below commented out as the clustering stage relies on ordering of merging fragments and
            #       grouping of fragments. Simply re-calling that function at this point with an updated
            #       angle calculation may not be a good idea (i.e. could merge 2 big showers by repeating
            #       merge fragments inside the reconstruct_shower_cluster function).
            #if self.cfg['modules']['shower_energy']['method'] == 'cone' and self.cfg['modules']['shower_match']['refit_cone']:
            #    self.reconstruct_shower_energy(event)
            
            
    def fiducialize(self, event): # TODO: Is this function needed anymore?
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
        if self.verbose:
            print(' Reconstructed pi0 masses: ', self.output['masses'])
        return masses


    def draw(self,**kargs):
        import plotly
        import numpy
        from mlreco.visualization.points import scatter_points
        import plotly.graph_objs as go
        from plotly.offline import iplot

        graph_data = []
        
        # Draw voxels with cluster labels
        # ------------------------------------
        energy = self.output['energy']
        shower_mask = self.output['shower_mask']
        graph_data += scatter_points(energy,markersize=2,color=energy[:,-1],colorscale='Inferno')
        graph_data[-1].name = 'Energy'
        
        
        # Add points from true electronShowers
        # ------------------------------------
        '''
        colors = plotly.colors.qualitative.Light24
        for i, s in enumerate(self.output['electronShowers']):
            if len(s.voxels)<1:
                continue
            color = colors[i % (len(colors))]
            graph_data += scatter_points(np.asarray(s.voxels),markersize=2,color=color)
            graph_data[-1].name = 'True electron shower %d (edep: %.2f)' %(i,s.edep_tot)
        
        if len(self.output['electronShowers'])>0:
            #points = np.array([s.start[0:3] for s in self.output['electronShowers']])
            #graph_data += scatter_points(points, markersize=3, color='deepskyblue')
            #graph_data[-1].name = 'True electron shower starts'

            points = np.array([s.first_step[0:3] for s in self.output['electronShowers']])
            graph_data += scatter_points(points, markersize=4, color='deepskyblue')
            graph_data[-1].name = 'True electron shower 1st steps'

            #points = np.array([s.first_edep[0:3] for s in self.output['electronShowers']])
            #graph_data += scatter_points(points, markersize=5, color='deepskyblue')
            #graph_data[-1].name = 'True electron shower 1st edeps'
        '''
        
        # Add points from true photonShowers
        # ------------------------------------
        '''
        colors = plotly.colors.qualitative.Light24
        for i, s in enumerate(self.output['photonShowers']):
            if len(s.voxels)<1:
                continue
            color = colors[(i+6) % (len(colors))]
            graph_data += scatter_points(np.asarray(s.voxels),markersize=2,color=color)
            graph_data[-1].name = 'True photon shower %d (edep: %.2f)' %(i,s.edep_tot)
        
        if len(self.output['photonShowers'])>0:
            #points = np.array([s.start[0:3] for s in self.output['photonShowers']])
            #graph_data += scatter_points(points, markersize=3, color='darkturquoise')
            #graph_data[-1].name = 'True photon shower starts'

            points = np.array([s.first_step[0:3] for s in self.output['photonShowers']])
            graph_data += scatter_points(points, markersize=4, color='darkturquoise')
            graph_data[-1].name = 'True photon shower 1st steps'

            #points = np.array([s.first_edep[0:3] for s in self.output['photonShowers']])
            #graph_data += scatter_points(points, markersize=5, color='darkturquoise')
            #graph_data[-1].name = 'True photon shower 1st edeps'
        '''
        
        # Add points from true comptonShowers
        # ------------------------------------
        '''
        colors = plotly.colors.qualitative.Light24
        for i, s in enumerate(self.output['comptonShowers']):
            if len(s.voxels)<1:
                continue
            color = colors[(i+12) % (len(colors))]
            graph_data += scatter_points(np.asarray(s.voxels),markersize=2,color=color)
            graph_data[-1].name = 'True compton shower %d (edep: %.2f)' %(i,s.edep_tot)
        
        if len(self.output['comptonShowers'])>0:
            #points = np.array([s.start[0:3] for s in self.output['comptonShowers']])
            #graph_data += scatter_points(points, markersize=3, color='darkcyan')
            #graph_data[-1].name = 'True compton shower starts'

            points = np.array([s.first_step[0:3] for s in self.output['comptonShowers']])
            graph_data += scatter_points(points, markersize=4, color='darkcyan')
            graph_data[-1].name = 'True compton shower 1st steps'

            #points = np.array([s.first_edep[0:3] for s in self.output['comptonShowers']])
            #graph_data += scatter_points(points, markersize=5, color='darkcyan')
            #graph_data[-1].name = 'True compton shower 1st edeps'
        '''
        
        # Add points from recoShowers
        # ------------------------------------
        #'''
        colors = plotly.colors.qualitative.Light24
        for i, s in enumerate(self.output['showers']):
            if len(s.voxels)<1:
                continue
            color = colors[(i+18) % (len(colors))]
            points = energy[shower_mask][s.voxels]
            graph_data += scatter_points(points,markersize=2,color=color)
            #graph_data[-1].name = 'Reco shower %d (n_edeps=%d, edep=%.2f, L_e=%.3f, L_p=%.3f, dir=[%.2f,%.2f,%.2f])' % (i,len(s.voxels),s.energy,s.L_e,s.L_p,s.direction[0],s.direction[1],s.direction[2])
            #graph_data[-1].name = 'Reco shower %d (edep: %.2f, dir: %.1f %.1f %.1f)' %(i,s.energy,s.direction[0],s.direction[1],s.direction[2])
            graph_data[-1].name = 'Reco shower %d (edep: %.2f, L_e=%.3f, L_p=%.3f)' %(i,s.energy,s.L_e,s.L_p)
        
        if len(self.output['showers'])>0:
            points = np.array([s.start for s in self.output['showers']])
            graph_data += scatter_points(points, markersize=3, color='gold')
            graph_data[-1].name = 'Reco shower starts'

            points = np.array([s.start for s in self.output['showers']])
            dirs = np.array([s.direction for s in self.output['showers']])
            cone_start = points[:,:3]
            arrows = go.Cone(x=cone_start[:,0], y=cone_start[:,1], z=cone_start[:,2],
                             u=-dirs[:,0], v=-dirs[:,1], w=-dirs[:,2],
                             sizemode='absolute', sizeref=1.0, anchor='tip',
                             showscale=False, opacity=0.4)
            graph_data.append(arrows)
        #'''
        
        
        # Add true pi0 decay points
        # ------------------------------------
        #'''
        if len(self.true_info['gamma_pos'])>0:
            true_pi0_decays = self.true_info['gamma_pos']
            graph_data += scatter_points(numpy.asarray(true_pi0_decays),markersize=5, color='green')
            for ind, vtx in enumerate(self.true_info['gamma_pos']):
                if ind%2 == 0:
                    graph_data[-1].name = 'True pi0 decay vertices (%.2f, %.2f, %.2f)' %(vtx[0],vtx[1],vtx[2])
        #'''
            
        # Add reconstructed pi0 decay points
        # ------------------------------------
        #'''
        try:
            reco_pi0_decays = self.output['vertices']
            graph_data += scatter_points(numpy.asarray(reco_pi0_decays),markersize=4, color='lightgreen')
            graph_data[-1].name = 'Reconstructed pi0 decay vertices'
        except:
            pass
        #'''
        
        # Add true photons 1st steps
        # ------------------------------------
        '''
        if len(self.true_info['gamma_first_step'])>0:
            true_gammas_first_steps = self.true_info['gamma_first_step']
            #print(' true_gammas_first_steps: ', true_gammas_first_steps)
            graph_data += scatter_points(numpy.asarray(true_gammas_first_steps), markersize=5, color='magenta')
            graph_data[-1].name = 'True photons 1st steps'
        '''
        
        # Add compton electrons 1st steps
        # ------------------------------------
        '''
        if len(self.true_info['compton_electron_first_step'])>0:
            compton_electrons_first_steps = self.true_info['compton_electron_first_step']
            #print(' compton_electrons_first_steps: ', compton_electrons_first_steps)
            graph_data += scatter_points(numpy.asarray(compton_electrons_first_steps), markersize=5, color='green')
            graph_data[-1].name = 'True compton electrons 1st steps'
        '''    
        # Add shower's true 1st (in time) step
        # ------------------------------------
        '''
        if len(self.true_info['shower_first_edep'])>0:
            shower_first_edep = self.true_info['shower_first_edep']
            #print(' shower_first_edep: ', shower_first_edep)
            graph_data += scatter_points(numpy.asarray(shower_first_edep), markersize=5, color='red')
            graph_data[-1].name = 'True showers 1st steps'
        '''

        # Add manually defined 3D points
        # ------------------------------------
        '''
        point_01 = np.array([469.86045002, 231.30654507, 514.07204156])
        #point_02 = np.array([406.88129432, 233.21140603, 107.01647391])
        #points = [np.asarray(point_01), np.asarray(point_02)] #,[325.2, 584.6, 312.3]]
        points = [np.asarray(point_01)]
        graph_data += scatter_points(numpy.asarray(points),markersize=4, color='orange')
        graph_data[-1].name = 'Manually defined point'
        
        point_02 = np.array([471.35858971, 244.42353517, 516.28956703])
        points = [np.asarray(point_02)]
        graph_data += scatter_points(numpy.asarray(points),markersize=4, color='lightgreen')
        graph_data[-1].name = 'Vtx candidate'
        '''

        # Add points predicted by PPN
        # ------------------------------------
        #'''
        if self.output['PPN_track_points']:
            points = np.array([i.ppns for i in self.output['PPN_track_points']])
            graph_data += scatter_points(points,markersize=4,color='magenta')
            graph_data[-1].name = 'PPN track points'
            #print(' PPN track points: ')
            #for i, point in enumerate(self.output['PPN_track_points']):
            #    print(' coord: ', point.ppns, ' \t track score: ', point.track_score, ' \t track id: ', point.track_id)
        if self.output['PPN_shower_points']:
            points = np.array([i.ppns for i in self.output['PPN_shower_points']])
            graph_data += scatter_points(points,markersize=4,color='purple')
            graph_data[-1].name = 'PPN shower points'
            #print(' PPN shower points: ')
            #for i, point in enumerate(self.output['PPN_shower_points']):
            #    print(' coord: ', point.ppns, ' \t shower score: ', point.shower_score, ' \t track id: ', point.shower_id)
        '''
        if self.output['PPN_michel_points']:
            points = np.array([i.ppns for i in self.output['PPN_michel_points']])
            graph_data += scatter_points(points,markersize=4,color='purple')
            graph_data[-1].name = 'PPN michel points'
        if self.output['PPN_delta_points']:
            points = np.array([i.ppns for i in self.output['PPN_delta_points']])
            graph_data += scatter_points(points,markersize=4,color='purple')
            graph_data[-1].name = 'PPN delta points'
        if self.output['PPN_LEScat_points']:
            points = np.array([i.ppns for i in self.output['PPN_LEScat_points']])
            graph_data += scatter_points(points,markersize=4,color='purple')
            graph_data[-1].name = 'PPN LEScatter points'
        '''

        # Add true photon's directions (based on true pi0 decay vertex and true photon's 1st steps)
        # ------------------------------------
        '''
        if 'gamma_pos' in self.true_info and 'gamma_first_step' in self.true_info:
            for i, true_dir in enumerate(self.true_info['gamma_pos']):
                vertex = self.true_info['gamma_pos'][i]
                first_step = self.true_info['gamma_first_step'][i]
                points = [vertex, first_step]
                graph_data += scatter_points(np.array(points),markersize=4,color='blue')
                #graph_data[-1].name = 'True photon %i vtx -> 1st step (einit: %.2f, edep: %.2f, 1st step: (%.2f,%.2f,%.2f))'\
                #                       %(i,self.true_info['gamma_ekin'][i],self.true_info['gamma_edep'][i],\
                #                         self.true_info['gamma_first_step'][i][0],self.true_info['gamma_first_step'][i][1],self.true_info['gamma_first_step'][i][2])
                graph_data[-1].name = 'True photon %i vtx -> 1st step' %i
                graph_data[-1].mode = 'lines,markers'
        '''
        
        # Add true photon's directions (based on true pi0 decay vertex and true photon's 1st (in time) edep)
        # ------------------------------------
        #'''
        if 'gamma_pos' in self.true_info and 'shower_first_edep' in self.true_info:
            for i, true_dir in enumerate(self.true_info['gamma_pos']):
                vertex = self.true_info['gamma_pos'][i]
                first_edep = self.true_info['shower_first_edep'][i]
                points = [vertex, first_edep]
                graph_data += scatter_points(np.array(points),markersize=4,color='green')
                #graph_data[-1].name = 'True photon %i: vtx -> 1st edep (einit: %.2f, edep: %.2f, 1st edep: (%.2f,%.2f,%.2f))'\
                #                       %(i,self.true_info['gamma_ekin'][i],self.true_info['gamma_edep'][i],\
                #                         self.true_info['shower_first_edep'][i][0],self.true_info['shower_first_edep'][i][1],self.true_info['shower_first_edep'][i][2])
                graph_data[-1].name = 'True photon %i: vtx -> 1st edep' %i
                graph_data[-1].mode = 'lines,markers'
        #'''

        # Add reconstructed pi0 decays, join vertex to start points
        # ------------------------------------
        #'''
        if 'matches' in self.output:
            for i, match in enumerate(self.output['matches']):
                v = self.output['vertices'][i]
                idx1, idx2 = match
                s1, s2 = self.output['showers'][idx1].start, self.output['showers'][idx2].start
                points = [v, s1, v, s2]
                graph_data += scatter_points(np.array(points),color='red')
                graph_data[-1].name = 'Reconstructed pi0 (%.2f MeV)' % self.output['masses'][i]
                graph_data[-1].mode = 'lines,markers'
        #'''

        # Add outer module dimensions (TODO: Check dimensions, probably add active volumes instead of outer module edges)
        # ------------------------------------
        #'''
        module_array = [2,2] # number of modules in x and y direction
        module_dimensions = [378, 378, 756] # x, y, z (in units of pixels)
        low_corners = []
        high_corners = []
        # Loop over all modules to obtain low and high corners
        for x_axis in range(module_array[0]):
            for y_axis in range(module_array[1]):
                low_corners.append([x_axis*module_dimensions[0], y_axis*module_dimensions[1], 0])
                high_corners.append([(x_axis+1)*module_dimensions[0], (y_axis+1)*module_dimensions[1], module_dimensions[2]])
        # Add module edges to graph_data
        for module_nr in range(len(low_corners)):
            points = []
            points.append([low_corners[module_nr][0],low_corners[module_nr][1],low_corners[module_nr][2]])
            points.append([low_corners[module_nr][0],low_corners[module_nr][1],high_corners[module_nr][2]])
            points.append([low_corners[module_nr][0],high_corners[module_nr][1],high_corners[module_nr][2]])
            points.append([low_corners[module_nr][0],high_corners[module_nr][1],low_corners[module_nr][2]])
            points.append([low_corners[module_nr][0],low_corners[module_nr][1],low_corners[module_nr][2]])
            points.append([high_corners[module_nr][0],low_corners[module_nr][1],low_corners[module_nr][2]])
            points.append([high_corners[module_nr][0],low_corners[module_nr][1],high_corners[module_nr][2]])
            points.append([high_corners[module_nr][0],high_corners[module_nr][1],high_corners[module_nr][2]])
            points.append([high_corners[module_nr][0],high_corners[module_nr][1],low_corners[module_nr][2]])
            points.append([high_corners[module_nr][0],low_corners[module_nr][1],low_corners[module_nr][2]])
            points.append([high_corners[module_nr][0],low_corners[module_nr][1],high_corners[module_nr][2]])
            points.append([low_corners[module_nr][0],low_corners[module_nr][1],high_corners[module_nr][2]])
            points.append([low_corners[module_nr][0],high_corners[module_nr][1],high_corners[module_nr][2]])
            points.append([high_corners[module_nr][0],high_corners[module_nr][1],high_corners[module_nr][2]])
            points.append([high_corners[module_nr][0],high_corners[module_nr][1],low_corners[module_nr][2]])
            points.append([low_corners[module_nr][0],high_corners[module_nr][1],low_corners[module_nr][2]])
            graph_data += scatter_points(np.array(points),color='#53EB83')
            graph_data[-1].mode = 'lines'
            graph_data[-1].name = 'Module %i' %module_nr
        #'''

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