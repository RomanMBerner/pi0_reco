import numpy as np
from larcv import larcv

class CSVData:

    def __init__(self, fout):
        self.name  = fout
        self._fout = None
        self._str  = None

    def write(self, dict):
        if self._str is None:
            self._fout = open(self.name,'w')
            self._str = ''
            for i,key in enumerate(dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str += '{}'
            self._fout.write('\n')
            self._str += '\n'
        self._fout.write(self._str.format(*(dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()

class Pi0DataLogger:

    def __init__(self, name):

        # Initialize log files
        self._label_log = CSVData(name+'_shower_label_data.csv')
        self._pred_log  = CSVData(name+'_shower_pred_data.csv')
        print('Will store true shower information at: '+name+'_shower_label_data.csv')
        print('Will store reconstructed shower information at: '+name+'_shower_pred_data.csv')

    def log(self, event, output):

        # Get the entry index
        eid = event['index']

        # Get the shower cluster fragment and group labels
        shower_mask = event['segment_label'][:,-1] == larcv.kShapeShower
        fragment_labels, group_labels = event['cluster_label'][:, 5], event['cluster_label'][:, 6]

        # Loop over the label shower groups, create an entry for each
        shower_label, shower_pixel_label, particle_label = [], [], []
        for i, g in enumerate(np.unique(group_labels[shower_mask])):
            # Initialize new shower entry with its entry index
            shower = {'event': eid, 'id': i}

            # Identify the group's first fragment in time and the group primary fragment
            group_mask = np.where(group_labels == g)[0]
            clust_ids  = np.unique(fragment_labels[group_mask]).astype(np.int64)
            times      = [event['particles'][i].first_step().t() for i in clust_ids]
            first_id   = clust_ids[np.argmin(times)]
            group_id   = event['particles'][first_id].group_id()
            first_p, group_p = event['particles'][first_id], event['particles'][group_id]

            # Record the first step of the first fragment in time (start point)
            shower['first_x'], shower['first_y'], shower['first_z'] =\
                first_p.first_step().x(), first_p.first_step().y(), first_p.first_step().z()

            # Record the normalized momentum of the primary fragment (direction)
            mom = np.array([group_p.px(), group_p.py(), group_p.pz()])
            shower['dir_x'], shower['dir_y'], shower['dir_z'] = np.array(mom)/np.linalg.norm(mom)

            # Record the voxel count of the group and integrate the energy
            shower['voxel_count'] = len(group_mask)
            shower['energy_int']  = np.sum(event['cluster_label'][group_mask, 4])

            # Record the requested list of label shower attributes
            for key in ['group_id', 'ancestor_track_id', 'pdg_code', 'parent_pdg_code',\
                        'ancestor_pdg_code', 'creation_process', 'energy_init', 'energy_deposit']:
                shower[key] = getattr(group_p, key)()

            # Append shower, record list of voxels associated with it
            shower_label.append(shower)
            shower_pixel_label.append(group_mask)
            particle_label.append(group_p)

        # Loop over the showers built by the chain
        shower_pred, shower_pixel_pred = [], []
        for i, s in enumerate(output.get('showers', [])):
            # Initialize new shower entry with its entry index
            shower = {'event': eid, 'id': i}

            # Record the reconstructed start and direction of the reconstructed shower
            shower['first_x'], shower['first_y'], shower['first_z'] = s.start
            shower['dir_x'], shower['dir_y'], shower['dir_z'] = s.direction

            # Record the voxel count and energy of the reconstructed shower
            shower['voxel_count'] = len(s.voxels)
            shower['energy_int'] = s.energy

            # Append shower, record list of voxels associated with it
            shower_pred.append(shower)
            shower_pixel_pred.append(s.voxels)

        # For each pair of one label and one reco shower, find the total number of shared voxels
        voxels_label, voxels_pred = event['input_data'][:,:3], output['charge'][:,:3]
        overlap_mat = np.empty((len(shower_label), len(shower_pred)))
        for i, spl in enumerate(shower_pixel_label):
            vl = voxels_label[spl]
            for j, spp in enumerate(shower_pixel_pred):
                vp = voxels_label[spp]
                overlap_mat[i,j] = np.sum([(v == vl).all(axis=1) for v in vp])

        # Match showers together (one match per true shower, one match per reco shower)
        for i, spl in enumerate(shower_pixel_label):

            # If there is no proposed shower, fill the default values
            if not len(shower_pixel_pred):
                shower_label[i]['match_id'] = -1
                shower_label[i]['match_overlap'] = 0
                continue

            # Store the matched id
            match_id = np.argmax(overlap_mat[i])
            shower_label[i]['match_id'] = match_id if overlap_mat[i, match_id] else -1
            shower_label[i]['match_overlap'] = overlap_mat[i, match_id]

        for i, spl in enumerate(shower_pixel_pred):

            # If there is no true shower, fill the default values
            if not len(shower_pixel_label):
                shower_pred[i]['match_id'] = -1
                shower_pred[i]['match_overlap'] = -1
                continue

            # Store the matched id
            match_id = np.argmax(overlap_mat[:,i])
            shower_pred[i]['match_id'] = match_id if overlap_mat[match_id, i] else -1
            shower_pred[i]['match_overlap'] = overlap_mat[match_id, i]

        # Match label showers into a pi0 by using their ancestor information
        for sl in shower_label:
            for key in ['id', 'x', 'y', 'z', 'angle', 'mass']:
                sl[f'pi0_{key}'] = -1

        anc_ids  = np.array([p.ancestor_track_id() for p in particle_label])
        # anc_pdgs = np.array([p.ancestor_pdg_code() for p in particle_label])
        par_pdgs = np.array([p.parent_pdg_code() for p in particle_label])
        for i, aid in enumerate(np.unique(anc_ids[par_pdgs == 111])):
            group_mask = np.where(anc_ids == aid)[0]
            if len(group_mask) < 2:
                continue

            for j in group_mask:
                # Store the pi0_label with a unique shower_id
                sl = shower_label[j]
                sl['pi0_id'] = i

                # Store the position of the decay vertex (pi0 production vertex)
                pos = particle_label[j].ancestor_position()
                sl['pi0_x'], sl['pi0_y'], sl['pi0_z'] = pos.x(), pos.y(), pos.z()

            # If there are exactly two showers, get the angle between photons and mass estimate
            if len(group_mask) == 2:
                sl1, sl2 = shower_label[group_mask[0]], shower_label[group_mask[1]]
                angle = np.arccos(np.dot((sl1['dir_x'], sl1['dir_y'], sl1['dir_z']),\
                                         (sl2['dir_x'], sl2['dir_y'], sl2['dir_z'])))
                mass  = np.sqrt(2.*sl1['energy_int']*sl2['energy_int']*(1-np.cos(angle)))
                sl1['pi0_angle'] = sl2['pi0_angle'] = angle
                sl1['pi0_mass']  = sl2['pi0_mass']  = mass

        # Match proposed photons into a pi0 by using distance of closest approach
        for sp in shower_pred:
            for key in ['id', 'x', 'y', 'z', 'angle', 'separation', 'mass']:
                sp[f'pi0_{key}'] = -1

        for i, match in enumerate(output.get('matches', [])):
            sp1, sp2 = shower_pred[match[0]], shower_pred[match[1]]
            angle    = np.arccos(np.dot((sp1['dir_x'], sp1['dir_y'], sp1['dir_z']),\
                                        (sp2['dir_x'], sp2['dir_y'], sp2['dir_z'])))
            for sp in [sp1, sp2]:
                # Store the reconstructed pi0 with a unique id
                sp['pi0_id'] = i

                # Store the position of the reconstructed decay vertex
                sp['pi0_x'], sp['pi0_y'], sp['pi0_z'] = output['vertices'][i]

                # Store the reconstructed angle between the two showers
                sp['pi0_angle'] = angle

                # Store the total angular disagreement between the two showers
                sp['pi0_separation'] = output['separations'][i]

                # Store the reconstructed mass
                sp['pi0_mass']  = output['masses'][i]

        # Store the data
        for sl in shower_label:
            self._label_log.write(sl)
            self._label_log.flush()
        for sp in shower_pred:
            self._pred_log.write(sp)
            self._pred_log.flush()
