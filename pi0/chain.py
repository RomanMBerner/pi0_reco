import yaml
from copy import copy
import numpy as np
from larcv import larcv

from mlreco.main_funcs import process_config, prepare, apply_event_filter
from mlreco.utils.ppn import uresnet_ppn_type_point_selector

from .utils.data_structure import Shower
from .utils.logger import Pi0DataLogger
from .analyse.analyser import Analyser

from .cluster.dbscan import DBSCANCluster
from .cluster.start_finder import StartPointFinder
from .cluster.cone_clusterer import ConeClusterer

from .directions.estimator import DirectionEstimator

from .identification.matcher import Pi0Matcher
from .identification.PID import ShowerIdentifier

from .visualization.plotting import draw_event


class Pi0Chain:
    '''
    Class that loads the Pi0 reconstruction configuration,
    intializes the chain modules and runs the inference.
    '''

    # Class constants
    IDX_CLUSTER_ID = 5
    IDX_GROUP_ID = 6

    def __init__(self, chain_cfg):
        '''
        Initialize the Pi0 reconstruction modules
        '''
        # Process chain configuration, enforce logic
        self._process_config(chain_cfg)

        # Initialize logger/analyser
        self._logger = Pi0DataLogger(self._name) # Francois'
        if self._analyse:
            self._analyser = Analyser() # Roman's

        # Initialize the fragment identifier
        if self._shower_fragment == 'dbscan':
            self._frag_est = DBSCANCluster(**self._shower_fragment_args)

        # Initliaze the start point finder
        if self._shower_start == 'curv':
            self._start_finder = StartPointFinder(**self._shower_start_args)

        # If a direction estimator is requested, initialize it
        if self._shower_direction != 'label':
            self._dir_est = DirectionEstimator(**self._shower_direction_args)

        # If a clusterer is requested, initialize it
        if self._shower_cluster != 'label':
            self._clusterer = ConeClusterer(**self._shower_cluster_args)

        if self._shower_id == 'edep' or self._shower_id == 'vertex':
            self._identifier = ShowerIdentifier(**self._shower_id_args)

        # If a pi0 identifier is requested, initialize it
        if self._shower_match == 'angle':
            self._matcher = Pi0Matcher(**self._shower_match_args)

        # Instantiate "handlers" (IO/inference tools)
        process_config(self._mlreco_cfg)
        self._hs = prepare(self._mlreco_cfg)
        self._data_set = iter(self._hs.data_io)

    def _process_config(self, chain_cfg):
        '''
        Processes the chain configuration file, enforces some basic logic
        between the modules used in the chain.
        '''
        # Process the configuration flags
        chain_cfg = yaml.safe_load(chain_cfg, )
        self._name    = chain_cfg.get('name', 'pi0_reco_chain')
        self._verbose = chain_cfg.get('verbose', False)
        self._analyse = chain_cfg.get('analyse', False)
        self._print('Initializing the Pi0 Reconstruction Chain...')

        # Initialize the pi0 reconstruction modules, save module parameters if any
        module_map = {'segment': ['label', 'uresnet'],
                      'deghost': ['label', 'uresnet', 'none'],
                      'charge2e': ['label', 'cst', 'average', 'none'],
                      'shower_fragment': ['label', 'dbscan', 'gnn'],
                      'shower_primary': ['label', 'gnn'],
                      'shower_start': ['label', 'curv', 'ppn'],
                      'shower_direction': ['label', 'geo'],
                      'shower_cluster': ['label', 'cone', 'gnn'],
                      'shower_energy': ['label', 'pixel_sum'],
                      'shower_id': ['label', 'edep', 'vertex', 'gnn', 'none'],
                      'shower_match': ['label', 'angle', 'gnn'],
                      'fiducial': ['none', 'edge_dist']}

        self._network = False
        assert 'modules' in chain_cfg, 'modules not specified in chain configuration'
        mod_cfg = chain_cfg['modules']
        self._print(f'Will run the following sequence of modules:')
        for module, keys in module_map.items():
            assert module in mod_cfg, f'{module} not specified under modules'
            assert 'method' in mod_cfg[module], f'method not specified under {module}'
            method = mod_cfg[module]['method']
            assert method in keys, f'{module} method not recognized: {method}. Must be one of: {keys}'
            setattr(self, f'_{module}', method)
            self._print(f' - {module:<20}: {method}')
            setattr(self, f'_{module}_args', {k:v for k, v in mod_cfg[module].items() if k != 'method'})
            if 'uresnet' or 'ppn' or 'gnn' in method: self._network = True

        # Enforce logical order of modules
        if self._shower_primary == 'gnn':
            assert self._shower_fragment == 'gnn', 'Shower fragments must be formed by GNN to identified by it'
        if self._shower_cluster == 'gnn':
            assert self._shower_primary == 'gnn', 'Shower primaries must be identified by GNN to be clustered by it'
        if self._shower_id == 'gnn':
            assert self._shower_cluster == 'gnn', 'Shower objects must be clustered by GNN for them to be identified by it'
        if self._shower_cluster != 'label':
            assert self._shower_energy != 'label', 'Label shower energy would ignore non-label clustering'

        # Initialize the mlreco configuration
        assert 'mlreco' in chain_cfg, 'mlreco not specified in chain configuration'
        mlreco_cfg = chain_cfg['mlreco']
        assert 'cfg_path' in mlreco_cfg, 'cfg_path not speicified under mlreco'
        with open(mlreco_cfg['cfg_path']) as cfg_file:
            self._mlreco_cfg = yaml.load(cfg_file, Loader=yaml.Loader)
            self._mlreco_cfg['iotool']['batch_size'] = mlreco_cfg.get('batch_size', 1)
            assert 'iotool' in self._mlreco_cfg, 'mlreco_cfg does not have an iotool entry'
            if 'sampler' in self._mlreco_cfg['iotool']:
                del self._mlreco_cfg['iotool']['sampler']
            if 'data_keys' in mlreco_cfg:
                self._mlreco_cfg['iotool']['dataset']['data_keys'] = mlreco_cfg['data_keys']
            if self._network:
                assert 'model' in self._mlreco_cfg and 'trainval' in self._mlreco_cfg
                self._mlreco_cfg['trainval']['train'] = False
                if 'model_path' in mlreco_cfg:
                    self._mlreco_cfg['trainval']['model_path'] = mlreco_cfg['model_path']

        # Initialize outputs
        self._output    = {}


    def _print(self, message):
        '''
        Only prints message in verbose mode.
        '''
        if self._verbose:
            print(message)


    def _log(self, event, output):
        '''
        Log event and output truth/reco information for further analysis.
        '''
        self._logger.log(event, output)


    @property
    def output(self):
        return self._output


    def apply_event_filter(self, entries):
        '''
        Narrow the dataset down to the entries specified.
        '''
        apply_event_filter(self._hs, entries)
        self._data_set = iter(self._hs.data_io)


    def get_fragment_labels(self, primary=False, group=False):
        '''
        Gets the most likely cluster ID/group ID of the shower fragments
        '''
        if primary:
            assert 'showers' in self._output
            fragments = [s.voxels for s in self._output['showers']]
        else:
            assert 'shower_fragments' in self._output
            fragments = self._output['shower_fragments']

        assert 'cluster_label' in self._output
        data = self._output['cluster_label']
        column = self.IDX_CLUSTER_ID if not group else self.IDX_GROUP_ID

        labels = []
        for f in fragments:
            v, cts = np.unique(data[f,column], return_counts=True)
            labels.append(int(v[np.argmax(cts)]))

        return np.array(labels)


    def get_ppn_track_points(self, event):
        '''
        Extracts PPN track point predictions from the raw PPN output.
        '''
        if 'ppn_track_points' not in self._output:
            from mlreco.utils.ppn import uresnet_ppn_type_point_selector
            points = uresnet_ppn_type_point_selector(event['input_data'],\
                {key: [self._output['forward'][key]] for key in ['segmentation', 'points', 'mask_ppn2']})
            point_sem = points[:,-1]
            ppn_track_points = points[point_sem == larcv.kShapeTrack, :3]
            self._output['ppn_track_points'] = ppn_track_points


    def run(self):
        '''
        Runs the full Pi0 reconstruction chain, from 3D charge
        information to Pi0 masses for events that contain one
        or more Pi0 decay.
        '''
        n_events = len(self._hs.data_io)
        for i in range(n_events):
            self.run_loop()


    def run_loop(self):
        '''
        Runs the full Pi0 reconstruction chain on a single event,
        from 3D charge information to Pi0 masses for events that
        contain one or more Pi0 decay.
        '''
        # Reset output
        self._output = {}

        # Load data
        if not self._network:
            event = next(self._data_set)
        else:
            event, self._output['forward'] = self._hs.trainer.forward(self._data_set)
            for key in event.keys():
                event[key] = event[key][0] # TODO: Assumes batch_size = 1
            for key in self._output['forward'].keys():
                self._output['forward'][key] = self._output['forward'][key][0] # TODO: Assumes batch_size = 1

        # Run the reconstruction modules
        self.run_modules(event)

        # Analyser module for reconstructed quantities
        # self._log(event, self._output)
        if self._analyse:
            self._analyser.record(event, self._output)


    def run_modules(self, event):
        '''
        Runs the reconstruction modules
        '''
        # Check input
        self.infer_inputs(event)

        # Set the semantics, abort if no shower voxels are found in the image
        self.infer_semantics(event)
        if not len(self._output['shower_mask']):
            self._print('No shower voxel found in event {}'.format(event['index']))
            return

        # Filter out ghosts, if necessary
        self.filter_ghosts(event)

        # Reconstruct energy
        self.charge_to_energy(event)

        # Form shower fragments, abort if there are no shower fragments
        self.reconstruct_shower_fragments(event)
        if not len(self._output['shower_fragments']):
            self._print('No shower fragment found in event {}'.format(event['index']))
            return

        # Identify primary fragments
        self.reconstruct_shower_primaries(event)
        if not len(self._output['showers']):
            self._print('No shower primary found in event {}'.format(event['index']))

        # Identify fragment start points
        self.reconstruct_shower_starts(event)

        # Reconstruct fragment direction vectors
        self.reconstruct_shower_directions(event)

        # Cluster shower fragments together into shower instances
        self.reconstruct_shower_cluster(event)

        # Reconstruct shower energy
        self.reconstruct_shower_energy(event)

        # Reconstruct shower likelihood ratios (electron/positron like or photon like)
        self.reconstruct_shower_id(event)

        # Identify which showers are contained within the fiducial volume.
        self.identify_fiducial(event)
        if len(self._output['showers']) < 2:
            self._print('Not enough showers found in event {} to form a pi0'.format(event['index']))
            return

        # Identify pi0 decays, abort mass reconstruction if no matches are found
        self.identify_pi0(event)
        if not len(self._output['matches']):
            self._print('No pi0 found in event {}'.format(event['index']))
            return

        # Compute masses
        self.pi0_mass()

    def infer_inputs(self, event):
        '''
        Ensures the data contains the necessary entries for the chain configuration.
        Copy the input as the event charge.
        '''
        if self._deghost == 'label' or self._segment == 'label':
            assert 'segment_label' in event, 'No segment_label data in the input, needed for true semantics'
            assert event['segment_label'].shape == event['input_data'].shape
            self._output['segment_label'] = copy(event['segment_label'])

        if self._charge2e == 'label':
            assert 'energy_label' in event, 'No energy_label data in the input, needed for true energy'
            assert event['energy_label' ].shape == event['input_data'].shape
            self._output['energy_label'] = copy(event['energy_label'])

        if 'label' in [self._shower_fragment, self._shower_primary,\
            self._shower_start, self._shower_direction, self._shower_cluster,\
            self._shower_energy, self._shower_id, self._shower_match]:
            assert 'cluster_label' in event
            assert event['cluster_label' ].shape[0] == event['input_data'].shape[0]
            self._output['cluster_label'] = copy(event['cluster_label'])

        if 'label' in [self._shower_primary, self._shower_start, self._shower_direction,\
                self._shower_energy, self._shower_id, self._shower_match]:
            assert 'particles' in event

        self._output['charge'] = copy(event['input_data'])


    def infer_semantics(self, event):
        '''
        Process the semantic classification followingn the configuration.
        Create a mask corresponding to the shower voxels.
        '''
        if self._segment == 'label':
            # The segmentation is the exact true segmentation
            self._output['segment'] = event['segment_label']

        elif self._segment == 'uresnet':
            # Get the segmentation output of the network, argmax to determine most probable label
            self._output['segment'] = copy(event['segment_label'])
            self._output['segment'][:,-1] = np.argmax(self._output['forward']['segmentation'], axis=1)

        self._output['shower_mask'] = np.where(self._output['segment'][:,-1] == larcv.kShapeShower)[0]


    def filter_ghosts(self, event):
        '''
        Removes ghost points from the charge tensor
        '''
        mask = None
        if self._deghost == 'none':
            # No deghosting needed, return
            return

        elif self._deghost == 'label':
            # Remove points labeled as ghosts
            mask = np.where(event['segment_label'][:,-1] != 5)[0]

        elif self._deghost == 'uresnet':
            # Get the segmentation output of the network, argmax to determine most probable label
            pred_ghost = np.argmax(self._output['forward']['ghost'], axis=1)
            mask = np.where(pred_ghost == 0)[0]

        self._output['charge']  = self._output['charge' ][mask]
        self._output['segment'] = self._output['segment'][mask]
        if 'energy_label' in self._output:  self._output['energy_label']  = self._output['energy_label'][mask]
        if 'segment_label' in self._output: self._output['segment_label'] = self._output['segment_label'][mask]
        if 'cluster_label' in self._output: self._output['cluster_label'] = self._output['cluster_label'][mask]

        self._output['shower_mask'] = np.where(self._output['segment'][:,-1] == larcv.kShapeShower)[0]


    def charge_to_energy(self, event):
        '''
        Reconstructs energy deposition from charge.
        '''
        if self._charge2e == 'none':
            # The input to the reconstruction chain is energy, ignore this step
            self._output['energy'] = copy(self._output['charge'])

        elif self._charge2e == 'label':
            # Use the energy_label as energy
            self._output['energy'] = self._output['energy_label']

        elif self._charge2e == 'constant':
            # Use a constant factor to convert from charge to energy
            assert 'cst' in self._charge2e_args, 'Constant factor for energy reconstruction not specified'
            self._output['energy'] = copy(self._output['charge'])
            self._output['energy'][:,-1] = self._charge2e_args['cst']*self._output['charge'][:,-1]

        elif self._charge2e == 'average':
            # Use an constant average energy value for all the voxels, ignore input charge
            assert 'average' in self._charge2e_args, 'Average energy per shower voxel not specified'
            self._output['energy'] = copy(self._output['charge'])
            self._output['energy'][:,-1] = self._charge2e_args['average']


    def reconstruct_shower_fragments(self, event):
        '''
        Cluster shower pixels into locally dense fragments
        '''
        if self._shower_fragment == 'label':
            # Loop over true fragments, append one fragment per unique true label
            fragments = []
            shower_label = self._output['cluster_label'][self._output['shower_mask']]
            for pid in np.unique(shower_label[:,self.IDX_CLUSTER_ID]):
                if pid > -1:
                    mask = np.where(shower_label[:,self.IDX_CLUSTER_ID] == pid)[0]
                    fragments.append(self._output['shower_mask'][mask])
            same_length = np.all([len(f) == len(fragments[0]) for f in fragments])
            self._output['shower_fragments'] = np.array(fragments, dtype=object if not same_length else np.int64)

        elif self._shower_fragment == 'dbscan':
            # Cluster shower energy depositions using dbscan
            shower_points    = self._output['energy'][self._output['shower_mask']]
            _, fragments, _  = self._frag_est.create_clusters(shower_points)
            same_length      = np.all([len(f) == len(fragments[0]) for f in fragments])
            self._output['shower_fragments'] = np.array([self._output['shower_mask'][f] for f in fragments], dtype=object if not same_length else np.int64)

        elif self._shower_fragment == 'gnn':
            # If the network already clustered the shower fragments, use as is
            assert self._network and 'shower_fragments' in self._output['forward'], 'shower_fragments not available in network output'
            self._output['shower_fragments'] = self._output['forward']['shower_fragments']

        # If any shower voxel is not a shower fragment, store as leftover energy
        full_mask = np.ones(len(self._output['segment']), dtype=np.bool)
        full_mask[np.concatenate(self._output['shower_fragments'])] = False
        full_mask = np.where(full_mask)[0]
        self._output['leftover_energy'] = full_mask[self._output['segment'][full_mask,-1] == larcv.kShapeShower]


    def reconstruct_shower_primaries(self, event):
        '''
        Identify fragments that initiated a shower (primaries). For each of the identified
        primary fragment, associate a unique shower object.
        '''
        if self._shower_primary == 'label':
            # Create one primary per shower group. Select the fragment with earliest time as primary
            clust_ids    = self.get_fragment_labels()
            group_ids    = self.get_fragment_labels(group=True)
            primary_mask = np.zeros(len(group_ids), dtype=bool)
            for g in np.unique(group_ids):
                mask = np.where(group_ids == g)[0]
                times = [event['particles'][i].first_step().t() for i in clust_ids[mask]]
                idx   = np.argmin(times)
                primary_mask[mask[idx]] = True

        elif self._shower_primary == 'gnn':
            # For each predicted shower group, pick the most likely node as the primary
            group_ids = self._output['forward']['shower_group_pred']
            if 'shower_node_pred' not in self._output['forward']:
                primary_mask = np.ones(1, dtype=bool)
            else:
                from scipy.special import softmax
                node_scores = softmax(self._output['forward']['shower_node_pred'], axis=1)
                primary_mask = np.zeros(len(group_ids), dtype=bool)
                for g in np.unique(group_ids):
                    mask = np.where(group_ids == g)[0]
                    idx  = node_scores[mask][:,1].argmax()
                    primary_mask[mask[idx]] = True

        # Create one shower object for each primary in the image, store leftover fragments
        self._output['showers'], self._output['leftover_fragments'] = [], []
        for i, f in enumerate(self._output['shower_fragments']):
            if primary_mask[i]:
                self._output['showers'].append(Shower(voxels=f, pid=group_ids[i]))
            else:
                self._output['leftover_fragments'].append(f)


    def reconstruct_shower_starts(self, event):
        '''
        Identify starting points of showers
        '''
        if self._shower_start == 'label':
            # For each primary fragment, use start point of the corresponding particle
            clust_ids = self.get_fragment_labels(primary=True)
            for i, s in enumerate(self._output['showers']):
                first_step = event['particles'][clust_ids[i]].first_step()
                s.start    = np.array([first_step.x(), first_step.y(), first_step.z()])

        elif self._shower_start == 'curv':
            # Use point of max umbrella curvature as starting point
            primary_frags = [s.voxels for s in self._output['showers']]
            start_points  = self._start_finder.find_start_points(self._output['energy'][:,:3], primary_frags)
            for i, s in enumerate(self._output['showers']):
                s.start = start_points[i]

        elif self._shower_start == 'ppn':
            # For each fragment, find the most likely point predicted *inside* the primary fragment
            from scipy.special import softmax
            voxels = self._output['energy'][:,:3]
            points_tensor = self._output['forward']['points']
            for i, s in enumerate(self._output['showers']):
                f       = s.voxels
                dmask   = np.where(np.max(np.abs(points_tensor[f,:3]), axis=1) < 1.)[0]
                scores  = softmax(points_tensor[f,3:5], axis=1)
                argmax  = dmask[np.argmax(scores[dmask,-1])] if len(dmask) else np.argmax(scores[:,-1])
                start   = voxels[f][argmax,:3] + points_tensor[f][argmax,:3] + 0.5
                s.start = start


    def reconstruct_shower_directions(self, event):
        '''
        Reconstructs the direction of the showers
        '''
        if self._shower_direction == 'label':
            # For each primary fragment, use the starting direction of the corresponding particle
            clust_ids = self.get_fragment_labels(primary=True)
            for i, s in enumerate(self._output['showers']):
                group_id    = event['particles'][clust_ids[i]].group_id()
                particle    = event['particles'][group_id]
                mom         = [particle.px(), particle.py(), particle.pz()]
                s.direction = np.array(mom)/np.linalg.norm(mom)

        elif self._shower_direction == 'geo':
            # Use a geometric method to estimate the direction of the primary shower fragment (PCA or centroid)
            starts     = np.array([s.start for s in self._output['showers']])
            fragments  = [self._output['energy'][s.voxels] for s in self._output['showers']]
            directions = self._dir_est.get_directions(starts, fragments)
            for i, shower in enumerate(self._output['showers']):
               shower.direction = directions[i]


    def reconstruct_shower_cluster(self,event):
        '''
        Cluster shower fragments and left-over pixels
        '''
        if self._shower_cluster == 'label':
            # Use group labels of shower fragments and leftover voxels to put them together
            frag_group_ids     = self.get_fragment_labels(group=True)
            primary_group_ids  = self.get_fragment_labels(primary=True, group=True)
            frag_mask          = np.ones(len(frag_group_ids), dtype=np.bool)
            for i, s in enumerate(self._output['showers']):
                group_mask = frag_group_ids == primary_group_ids[i]
                s.voxels   = np.concatenate(self._output['shower_fragments'][group_mask])
                frag_mask[group_mask] = False

            lo_mask = np.ones(len(self._output['leftover_energy']), dtype=np.bool)
            for i, s in enumerate(self._output['showers']):
                group_mask = self._output['cluster_label'][self._output['leftover_energy'],self.IDX_GROUP_ID] == primary_group_ids[i]
                s.voxels = np.concatenate([s.voxels, self._output['leftover_energy'][group_mask]])
                lo_mask[group_mask] = False

            self._output['leftover_fragments'] = self._output['shower_fragments'][frag_mask]
            self._output['leftover_energy'] = self._output['leftover_energy'][lo_mask]

        elif self._shower_cluster == 'cone':
            # Build one cone per starting point, merge fragments and leftovers
            # self.merge_fragments(event) # TODO: This module is defective, seems to output a lot of crap, must investigate
            self.merge_leftovers(event)

        elif self._shower_cluster == 'gnn':
            # Use the GNN group predictions to cluster leftover fragments
            primary_group_ids = [s.pid for s in self._output['showers']]
            frag_group_ids    = self._output['forward']['shower_group_pred']
            frag_mask         = np.ones(len(frag_group_ids), dtype=np.bool)
            for i, s in enumerate(self._output['showers']):
                group_mask = frag_group_ids == primary_group_ids[i]
                s.voxels   = np.concatenate(self._output['shower_fragments'][group_mask])
                frag_mask[group_mask] = False

            self._output['leftover_fragments'] = self._output['shower_fragments'][frag_mask]

            # TODO: Add option to merge leftover energy using cones ? Ignore for now

    def merge_fragments(self, event):
        '''
        Merge shower fragments with assigned start point
        '''
        from pi0.cluster.fragment_merger import group_fragments
        impact_parameter = float(self._shower_cluster_args['IP'])
        radiation_length = float(self._shower_cluster_args['Distance'])
        points = self._output['energy']
        fragments = []
        for i, s in enumerate(self._output['showers']):
            start = s.start
            voxel = points[s.voxels]
            fragments.append([start,voxel])

        roots, groups, pairs = group_fragments(fragments, dist_prep=impact_parameter, dist_rad=radiation_length)
        assert(len(roots) == len(groups))
        # loop over groups and merge fragments
        showers = []
        fragments = []
        for idx,root in enumerate(roots):
            # merge secondaries
            showers.append(self._output['showers'][root])
            secondaries = [self._output['showers'][fidx].voxels for fidx in groups[idx]]
            showers[-1].voxels = np.concatenate(secondaries)
        self._output['showers'] = showers


    def merge_leftovers(self, event):
        '''
        Merge leftover fragments (w/o start point) and leftover pixels
        '''
        # Fits cones to each shower, adds energies within that cone
        starts = np.array([s.start for s in self._output['showers']])
        dirs = np.array([s.direction for s in self._output['showers']])
        remaining_inds = np.concatenate(self._output['leftover_fragments'] + [self._output['leftover_energy']]).astype(np.int32)
        if len(remaining_inds) < 1:
            return

        energy = self._output['energy']
        remaining_energy = energy[remaining_inds]
        fragments = [energy[s.voxels] for s in self._output['showers']]
        pred = self._clusterer.fit_predict(remaining_energy, starts, fragments, dirs)

        for i, s in enumerate(self._output['showers']):
            merging_inds = remaining_inds[np.where(pred == i)]
            s.voxels = np.concatenate([s.voxels,merging_inds])


    def reconstruct_shower_energy(self, event):
        '''
        Reconstruct the energy of the showers
        '''
        if self._shower_energy == 'label':
            # Get the true total shower energy from the particle tree
            clust_ids = self.get_fragment_labels(primary=True)
            for i, s in enumerate(self._output['showers']):
                group_id = event['particles'][clust_ids[i]].group_id()
                s.energy = event['particles'][group_id].energy_init()

        elif self._shower_energy == 'pixel_sum':
            # Sum the energy of all the voxels in the shower
            for s in self._output['showers']:
                s.energy = np.sum(self._output['energy'][s.voxels][:,-1])
                if 'fudge' in self._shower_energy_args:
                    s.energy *= self._shower_energy_args['fudge']


    def reconstruct_shower_id(self, event):
        '''
        Obtain the electron/positron and photon likelihood ratios for the showers.
        '''
        if self._shower_id == 'none':
            # Set all photon likelihood to one, all the showers will be candidates to be matched
            for i, s in enumerate(self._output['showers']):
                s.L_e = 0.
                s.L_p = 1.

        if self._shower_id == 'label':
            # Use the true particle ID to determine whether the ancestor is a photon or an electron
            clust_ids = self.get_fragment_labels(primary=True)
            for i, s in enumerate(self._output['showers']):
                particle = event['particles'][clust_ids[i]]
                anc_pdg  = particle.ancestor_pdg_code()
                s.L_e = int(abs(anc_pdg) == 11)
                s.L_p = int(anc_pdg == 22 or anc_pdg == 111 or (anc_pdg == 0 and particle.parent_pdg_code() == 22))

        elif self._shower_id == 'edep':
            # Use the energy deposition at the start of a shower a criterion for e/gamma separation
            # TODO: This method is inherently flawed. If a gamma compton scatters, the primary is very much an electron (way around that ?)
            self._identifier.set_edep_lr(self._output['showers'], self._output['energy'])

        elif self._shower_id == 'vertex':
            # Uses the proximity to a vertex (PPN track point) as a criterion for e/gamma separation
            self.get_ppn_track_points(event)
            self._identifier.set_vertex_lr(self._output['showers'], self._output['ppn_track_points'])

        elif self._shower_id == 'gnn':
            # Use the GNN to preidict its node types (electron or photon shower)
            assert 'inter_node_pred' in self._output['forward'], 'Need node predictions in the interaction GNN to do PID with it'
            shower_mask = np.where(self._output['forward']['particles_seg'] == larcv.kShapeShower)[0]
            assert len(shower_mask) == len(self._output['showers'])

            from scipy.special import softmax
            node_scores = softmax(self._output['forward']['inter_node_pred'], axis=1)
            for i, s in enumerate(self._output['showers']):
                s.L_e = node_scores[shower_mask[i], 1]
                s.L_p = node_scores[shower_mask[i], 0]


    def identify_fiducial(self, event):
        '''
        If a shower has energy depositions outside of fiducial volume,
        record it as a shower attribute
        '''
        if self._fiducial == 'none':
            # If no fiducial cut is required, skip this step
            pass

        elif self._fiducial == 'edge_dist':
            # Loop over showers, check if any of the energy deposits are outside of fiducial
            assert 'max_distance' in self._fiducial_args, 'Need to specify minimum distance from volume edge'
            max_dist = self._fiducial_args.get('max_distance')
            lower    = self._fiducial_args.get('lower_bound', 0)
            upper    = self._fiducial_args.get('upper_bound', 768)

            for s in self._output['showers']:
                coords = self._output['energy'][s.voxels,:3]
                if np.any(coords < (lower+max_dist)) or np.any(coords >= (upper-max_dist)):
                    s.fiducial = False


    def identify_pi0(self, event):
        '''
        Proposes pi0 candidates (match pairs of showers)
        '''
        self._output['matches']  = []
        self._output['vertices'] = []
        n_showers = len(self._output['showers'])

        if self._shower_match == 'label':
            # Make the pairs based on ancestor track id, only consider 111 ancestors
            clust_ids     = self.get_fragment_labels(primary=True)
            anc_ids, pdgs = [], []
            for pid in clust_ids:
                particle = event['particles'][pid]
                anc_ids.append(int(particle.ancestor_track_id()))
                pdgs.append(particle.ancestor_pdg_code())

            anc_ids, pdgs  = np.array(anc_ids), np.array(pdgs)
            pi0_mask = np.where(pdgs == 111)[0]
            for aid in np.unique(anc_ids[pi0_mask]):
                group_mask = np.where(anc_ids == aid)[0]
                if len(group_mask) < 2: continue
                if len(group_mask) > 2:
                    sizes = [len(primary_frags[i]) for i in group_mask]
                    order = np.argsort(sizes)
                    group_mask = group_mask[order][-2:]

                pos = event['particles'][group_mask[0]].ancestor_position()
                self._output['matches'].append(group_mask)
                self._output['vertices'].append(np.array([pos.x(), pos.y(), pos.z()]))

        elif self._shower_match == 'angle':
            # If the matcher needs PPN, extract the PPN track points
            if self._matcher._match_to_ppn: self.get_ppn_track_points(event)

            # Pair showers which are most likely to originate from a common vertex
            track_mask = self._output['segment'][:,-1] == larcv.kShapeTrack
            self._output['matches'], self._output['vertices'], _ =\
                self._matcher.find_matches(self._output['showers'],
                                           self._output['segment'][track_mask, :3],
                                           self._output.get('ppn_track_points'))

        elif self._shower_match == 'gnn':
            raise NotImplementedError('Will be able to use interaction clustering to orient Pi0 pairings')

        # If requested, use the newly reconstructed Pi0 vertex to adjust the shower directions
        if self._shower_match_args.get('refit_dir', False):
            for i, match in enumerate(self._output['matches']):
                idx1, idx2 = match
                v = np.array(self._output['vertices'][i])
                for shower_idx in match:
                    new_dir = self._output['showers'][shower_idx].start - v
                    if not np.linalg.norm(new_dir):
                        self._print(f'Direction of shower {shower_idx} was not refitted because its start point coincides the pi0 vertex')
                        continue
                    # Only take new direction if shower start position is not too close to the vertex
                    # TODO: Optimize this parameter
                    if np.linalg.norm(new_dir) < 5.:
                        continue
                    self._output['showers'][shower_idx].direction = new_dir/np.linalg.norm(new_dir)

        if self._shower_energy == 'cone' and self._shower_match_args.get('refit_cone', False):
            self.reconstruct_shower_energy(event)


    def pi0_mass(self):
        '''
        Reconstructs the pi0 mass
        '''
        self._output['masses'] = []
        for match in self._output['matches']:
            idx1, idx2 = match
            s1, s2 = self._output['showers'][idx1], self._output['showers'][idx2]
            e1, e2 = s1.energy, s2.energy
            t1, t2 = s1.direction, s2.direction
            costheta = np.dot(t1, t2)
            self._output['masses'].append(np.sqrt(2.*e1*e2*(1.-costheta)))

    def draw(self, **kwargs):
        """
        Draws the event processed in the last run_loop.
        """
        #draw_event(self._output, self._analyser.true_info, **kwargs)
        draw_event(self._output, None, **kwargs)
