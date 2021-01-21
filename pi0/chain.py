import numpy as np
import yaml
from copy import copy
from larcv import larcv
from .directions.estimator import FragmentEstimator, DirectionEstimator
from .cluster.start_finder import StartPointFinder
from .cluster.cone_clusterer import ConeClusterer
from .cluster.dbscan import DBSCANCluster
from .identification.matcher import Pi0Matcher
from .utils.logger import Pi0DataLogger
from mlreco.main_funcs import process_config, prepare

# Class that contains all the shower information
class Shower:
    def __init__(self, start=-np.ones(3), direction=-np.ones(3), voxels=[], energy=-1., pid=-1, group_pred=-1, L_e=-1., L_p=-1.):
        self.start      = start
        self.direction  = direction
        self.voxels     = voxels
        self.energy     = energy
        self.pid        = int(pid)
        self.group_pred = int(group_pred)
        self.L_e        = L_e # electron (positron) likelihood fraction
        self.L_p        = L_p # photon likelihood fraction

    def __str__(self):
        return """
        Shower ID  : {}
        Start point: ({:0.2f},{:0.2f},{:0.2f})
        Direction  : ({:0.2f},{:0.2f},{:0.2f})
        Voxel count: {}
        Energy     : {}
        Group_pred : {}
        """.format(self.pid, *self.start, *self.direction, len(self.voxels), self.energy, self.group_pred)


# Chain object class that loads and stores the chain parameters
class Pi0Chain:

    # Class constants
    IDX_SEMANTIC_ID = -1
    IDX_CLUSTER_ID = 5
    IDX_GROUP_ID = 6

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
        print('Initialized Pi0 reconstruction chain')
        self._log = Pi0DataLogger(chain_cfg['name'])

        # If a network is specified, initialize the network
        self.network = False
        if chain_cfg['segment'] == 'uresnet' or chain_cfg['shower_start'] == 'ppn' or chain_cfg['shower_start'] == 'gnn':
            self.network = True
            with open(chain_cfg['net_cfg']) as cfg_file:
                net_cfg = yaml.load(cfg_file,Loader=yaml.Loader)
            io_cfg['model'] = net_cfg['model']
            io_cfg['trainval'] = net_cfg['trainval']

        # Initialize the fragment identifier
        if chain_cfg['shower_fragment'] == 'dbscan':
            self.frag_est = DBSCANCluster()

        # If a direction estimator is requested, initialize it
        if chain_cfg['shower_direction'] != 'label':
            self.dir_est = DirectionEstimator()

        # If a clusterer is requested, initialize it
        if chain_cfg['shower_cluster'] in ['cone', 'gnn']:
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

    def log(self, event, output):
        self._log.log(event, output)

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

        # Filter out ghosts
        self.filter_ghosts(event)

        # Reconstruct energy
        self.charge_to_energy(event)

        # Check data dimensions
        assert self.output['energy'].shape == self.output['charge'].shape
        assert self.output['energy'].shape == self.output['segment'].shape

        # Identify shower starting points, skip if there is less than 2 (no pi0)
        self.reconstruct_shower_starts(event)
        if len(self.output['showers']) < 2:
            if self.verbose:
                print('No shower start point found in event', event_id)
            return

        # Form shower fragments
        self.reconstruct_shower_fragments(event)

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
            #self.log(self.event, self.output)
            return

        # Compute masses
        masses = self.pi0_mass()

        # Log data
        #self.log(self.event, self.output)


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

        elif self.cfg['shower_start'] == 'ppn':
            from mlreco.utils.ppn import uresnet_ppn_type_point_selector
            shower_score_index = -1 * (int(larcv.kShapeUnknown) - int(larcv.kShapeShower))
            point_score_index  = -1 * (int(larcv.kShapeUnknown) + 1)
            points = uresnet_ppn_type_point_selector(self.event['input_data'], self.output['forward'])
            points = points[np.where(points[:,shower_score_index] > self.cfg.get('shower_score_threshold',0.5))]
            total_score = points[:,shower_score_index] * points[:,point_score_index]
            order  = np.argsort(total_score)
            self.output['showers'] = [Shower(start=points[i,:-1],pid=int(i)) for i in order]

        elif self.cfg['shower_start'] == 'gnn':
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
            primary_clusts_group_pred = []
            for i, primary_cluster in enumerate(primary_clusts):
                for j, all_cluster in enumerate(all_clusts):
                    if all_cluster[0]==primary_cluster[0]:
                        primary_clusts_group_pred.append(group_ids[j])
                        break

            # Getting start points from GNN
            start_finder = StartPointFinder()
            start_points = start_finder.find_start_points(self.output['energy'][:,:3], primary_clusts)
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
            clusts, remaining_clusts, remaining_energy = self.frag_est.create_clusters(shower_points, shower_starts)
            showers = []
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
            try:
                res = self.dir_est.get_directions(starts, fragments, max_distance=float('inf'), mode=algo)
            except AssertionError as err: # Cluster was not found for at least one primary
                if self.verbose:
                    print('Error in direction reconstruction:', err)
                res = [[0., 0., 0.] for _ in range(len(self.output['showers']))]

            for i, shower in enumerate(self.output['showers']):
                shower.direction = res[i]

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
            group_ids = self.output['forward']['shower_group_pred'][0]
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
        self.output['matches'] = []
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
                        print(events['particles'][0][p].dump())

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
                for i, match in enumerate(self.output['matches']):
                    idx1, idx2 = match
                    v = np.array(self.output['vertices'][i])
                    for shower_idx in [idx1,idx2]:
                        new_dir = np.array(points[shower_idx]) - v
                        if np.linalg.norm(new_dir) > 5:
                            self.output['showers'][shower_idx].direction = new_dir/np.linalg.norm(new_dir)

            # Below commented out as the clustering stage relies on ordering of merging fragments and
            #       grouping of fragments. Simply re-calling that function at this point with an updated
            #       angle calculation may not be a good idea (i.e. could merge 2 big showers by repeating
            #       merge fragments inside the reconstruct_shower_cluster function).
            #if self.cfg['shower_energy'] == 'cone' and self.cfg['refit_cone']:
            #    self.reconstruct_shower_energy(event)

        else:
            raise ValueError('Shower matching method not recognized:', self.cfg['shower_match'])

    def pi0_mass(self):
        '''
        Reconstructs the pi0 mass
        '''
        from math import sqrt
        masses = []
        for match in self.output['matches']:
            idx1, idx2 = match
            s1, s2 = self.output['showers'][idx1], self.output['showers'][idx2]
            e1, e2 = s1.energy, s2.energy
            t1, t2 = s1.direction, s2.direction
            costheta = np.dot(t1, t2)
            if abs(costheta) > 1.:
                masses.append(0.)
                continue
            masses.append(sqrt(2*e1*e2*(1-costheta)))
        self.output['masses'] = masses
        return masses

    def draw(self,**kargs):
        import plotly
        from mlreco.visualization.points import scatter_points
        import plotly.graph_objs as go
        from plotly.offline import iplot

        graph_data = []
        # Draw voxels with cluster labels
        energy = self.output['energy']
        shower_mask = self.output['shower_mask']
        graph_data += scatter_points(energy,markersize=2,color=energy[:,-1],colorscale='Inferno')
        graph_data[-1].name='Energy'

        colors = plotly.colors.qualitative.Light24
        for i, s in enumerate(self.output['showers']):
            points = energy[shower_mask][s.voxels]
            color = colors[i % (len(colors))]
            graph_data += scatter_points(points,markersize=2,color=color)
            graph_data[-1].name = 'Shower %d (id=%d)' % (i,s.pid)

        if len(self.output['showers']):

            # Add EM primary points
            points = np.array([s.start for s in self.output['showers']])
            graph_data += scatter_points(points)
            graph_data[-1].name = 'Shower Starts'

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
                    graph_data[-1].name = 'Pi0 (%.2f MeV)' % self.output['masses'][i]
                    graph_data[-1].mode = 'lines,markers'

        # Draw
        #go.Figure(data=graph_data,layout=self.layout(**kargs))
        from mlreco.visualization import plotly_layout3d
        fig = go.Figure(data=graph_data,layout=plotly_layout3d())
        camera = dict(
            up=dict(x=0, y=0, z=0),
            center=dict(x=0, y=0, z=-0.3),
            eye=dict(x=0.4, y=0.4, z=0.4)
        )
        fig.update_layout(scene_camera=camera, showlegend=False)
        iplot(fig)
