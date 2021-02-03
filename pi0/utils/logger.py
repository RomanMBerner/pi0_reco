import numpy as np
from larcv import larcv
from scipy.spatial.distance import cdist
from pi0.identification.matcher import Pi0Matcher

class CSVData:

    def __init__(self,fout):
        self.name  = fout
        self._fout = None
        self._str  = None
        self._dict = {}

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def write(self):
        if self._str is None:
            self._fout=open(self.name,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'
        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()

class Pi0DataLogger:

    def __init__(self, name):

        # Initialize log files
        self.label_log = CSVData(name+'_shower_label_data.csv')
        self.pred_log = CSVData(name+'_shower_pred_data.csv')
        print('Will store true shower information at: '+name+'_shower_label_data.csv')
        print('Will store reconstructed shower information at: '+name+'_shower_pred_data.csv')

    def log(self, event, output):
        # Get the entry index
        eid = event['index']

        # Get an array for each truth and predictions
        print(event['particles'][0])
        for p in event['particles'][0]:
            print(p.pdg_code(), p.shape(), p.id(), p.group_id())
        primary_showers = [p for p in event['particles'][0]\
                if p.shape == larcv.kShapeShower and p.id() == p.group_id()]
        reco_showers = [] if 'showers' not in output else output['showers']

        # Loop over the true showers
        shower_label = []
        shower_pixel_label = []
        for p in primary_showers:
            shower = {}
            shower['pos_x'], shower['pos_y'], shower['pos_z'] = p.position().x(), p.position.y(), p.position.z()
            shower['first_x'], shower['first_y'], shower['first_z'] = p.first_step().x(), p.first_step.y(), p.first_step.z()

            norm = np.linalg.norm((p.momentum().x(), p.momentum().y(), p.momentum().z()))
            shower['dir_x'], shower['dir_y'], shower['dir_z'] = p.momentum().x()/norm, p.momentum().y()/norm, p.momentum().z()/norm
            shower['energy_init'] = p.energy_init()
            shower['energy_deposit'] = p.energy_deposit()

            mask = event['cluster_label'][0][:,6] == p.group_id()
            shower['voxel_count'] = np.sum(mask)
            shower['energy_int'] = np.sum(event['cluster_label'][0][mask,4])

            shower['event'] = eid
            shower['id'] = p.id()
            shower['group_id'] = p.groud_id()
            shower['ancestor_id'] = p.ancestor_track_id()
            shower['pdg_code'] = p.pdg_code()
            shower['parent_pdg_code'] = p.parent_pdg_code()

            shower_label.append(shower)
            shower_pixel_label.append(np.where(mask))

        # Loop over the showers built by the chain
        shower_pred = []
        shower_pixel_pred = []
        for i, s in enumerate(reco_showers):
            shower = {}
            shower['first_x'], shower['first_y'], shower['first_z'] = s.start
            shower['dir_x'], shower['dir_y'], shower['dir_z'] = s.direction

            shower_mask = output['shower_mask'][0]
            shower['voxel_count'] = len(s.voxels)
            shower['energy_int'] = s.energy

            shower['event'] = eid
            shower['id'] = i

            shower_pred.append(shower)
            shower_pixel_pred.append(shower_mask[s.voxels])

        print(shower_pred)

        # Match showers together (one match per true shower, one match per reco shower)
        voxels = output['energy'][:,:3]
        dist_mat = np.full((len(shower_label), len(shower_pred)), 1e9)
        for i, spl in enumerate(shower_pixel_label):

            # If there is no proposed shower, fill the default values
            if not len(shower_pixel_pred):
                shower_label[i]['match_id'] = -1
                shower_label[i]['match_dist'] = -1
                shower_label[i]['match_overlap'] = -1

            # Find the shower that is closest to the true point set
            for j, spp in enumerate(shower_pixel_pred):
                if i > j:
                    dist_mat[i,j] = np.min(cdist(voxels[spl], voxels[spp]))
                elif j < i:
                    dist_mat[i,j] = dist_mat[j,i]

            # Store the matched id
            match_id = np.argmin(dist_mat[i])
            shower_label[i]['match_id'] = match_id
            shower_label[i]['match_dist'] = dist_mat[i, match_id]

            # Store the pixel overlap
            spp = shower_pixel_pred[match_id]
            shower_label[i]['match_overlap'] = np.sum([(spl == k).any() for k in spp])

        for i, spl in enumerate(shower_pixel_pred):

            # If there is no true shower, fill the default values
            if not len(shower_pixel_label):
                shower_pred[i]['match_id'] = -1
                shower_pred[i]['match_dist'] = -1
                shower_pred[i]['match_overlap'] = -1

            # Store the matched id
            match_id = np.argmin(dist_mat[:,i])
            shower_label[i]['match_id'] = match_id
            shower_label[i]['match_dist'] = dist_mat[match_id, pred]

            # Store the pixel overlap
            spl = shower_pixel_label[match_id]
            shower_pred[i]['match_overlap'] = np.sum([(spl == k).any() for k in spp])


        # Match true photons into a pi0 by using parentage information
        for i, sl1 in shower_label:
            found_match = False
            for j, sl2 in shower_label:
                if i != j and sl1['parent_pdg_code'] == 111 and sl2['parent_pdg_code'] == 111 and\
                        sl1['ancestor_id'] == sl2['ancestor_id']:
                    # IS THAT GOOD ENOUGH (NEED TO CHECK, TODO)
                    # Store the vertex
                    found_match = True
                    sl1['pi0_x'], sl1['pi0_y'], sl1['pi0_z'] = sl1['pos_x'], sl1['pos_y'], sl1['pos_z']

                    # Store the relative angle between photons (dot product)
                    sl1['pi0_angle'] = np.dot((sl1['dir_x'], sl1['dir_y'],\
                            sl1['dir_z']),(sl2['dir_x'], sl2['dir_y'], sl2['dir_z']))

                    # Store the mass
                    sl1['pi0_mass'] = np.sqrt(2.*sl1['energy_int']*sl2['energy_int']*(1-sl1['pi0_angle']))
                if not found_match:
                    sl1['pi0_x'], sl1['pi0_y'], sl1['pi0_z'] = -1, -1, -1
                    sl1['pi0_angle'] = -1
                    sl1['pi0_mass'] = -1

        # Match proposed photons into a pi0 by using distance of closest approach
        for i, s1 in reco_showers:

            # If there is only one shower, keep going
            sp = shower_pred[i]
            if len(reco_showers) < 2:
                sp['pi0_x'], sp['pi0_y'], sp['pi0_z'] = -1, -1, -1
                sp['pi0_angle'] = -1
                sp['pi0_mass'] = -1

            # Compute distance between points of closests approach for all combinations
            dists, vertices = [], []
            for j, s2 in reco_showers:
                # Find the points of closest approach
                if i == j:
                    continue
                pocas = Pi0Matcher.find_pocas(np.vstack(s1.start, s2.start),\
                                 np.vstack(s1.direction, s2.direction))

                # Store the distance the midpoint (vertex approx)
                dists.append(np.linalg.norm(pocas[1]-pocas[0]))
                verticed.append((pocas[0]+pocas[1])/2)

            # Select the closest shower, store all the info
            match_id = np.argmin(dists)
            sp['pi0_x'], sp['pi0_y'], sp['pi0_z'] = vertices[match_id]
            sp['pi0_sep'] = dists[match_id]

            # Store the relative angle between photons (dot product)
            sp['pi0_angle'] = np.dot(s1.direction, reco_showers[match_id].direction)

            # Store the mass
            sp['pi0_mass'] = np.sqrt(2.*s1.energy*s2.energy*(1-sp['pi0_angle']))

        # Store the data
        self.label_log.record(shower_label.keys(), shower_label.values())
        self.label_log.write()
        self.label_log.flush()
        self.pred_log.record(shower_pred.keys(), shower_pred.values())
        self.pred_log.write()
        self.pred_log.flush()
