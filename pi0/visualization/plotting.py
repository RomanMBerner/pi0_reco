import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot
from mlreco.visualization.points import scatter_points

def layout(width=1024, height=768, xrange=(0,768), yrange=(0,768), zrange=(0,768), dark=False, aspectmode='cube'):
    '''
    Returns a default layout for the Pi0 reconstruction output
    '''
    layout = go.Layout(
        showlegend = True,
        legend     = dict(x=0.95,y=0.95),
        width      = width,
        height     = height,
        hovermode  = 'closest',
        margin     = dict(l=0,r=0,b=0,t=0),
        uirevision = 'same',
        scene      = dict(xaxis = dict(nticks=10, range = xrange, showticklabels=True, title='x'),
                          yaxis = dict(nticks=10, range = yrange, showticklabels=True, title='y'),
                          zaxis = dict(nticks=10, range = zrange, showticklabels=True, title='z'),
                          aspectmode=aspectmode)
    )
    if dark: layout.template = 'plotly_dark'

    return layout


def high_contrast_colorscale(n_classes=-1, gray_start=False):
    '''
    Defines a colorscale that supports up to 48 labels with sharp color changes
    between successive values. Ideal to plot integers values.
    '''
    import plotly.express as px
    colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    if gray_start: colors = ['#E2E2E2'] + colors
    assert n_classes <= (48 + int(gray_start))
    if n_classes < 1: n_classes = (48 + int(gray_start))
    step = 1./n_classes
    colorscale = []
    for i in range(n_classes):
        colorscale.append([i*step, colors[i]])
        colorscale.append([(i+1)*step, colors[i]])

    return colorscale


def module_boxes(arr=[2,2], size=[378,378,756]):
    '''
    Produces subdector modules of dimensions [dx,dy,dz] in a [n_x,n_y] arrangement
    '''
    # Loop over all modules to obtain the lower extremity for each module
    lowers = np.vstack([np.array([i*size[0], j*size[1], 0]) for i in range(arr[0]) for j in range(arr[1])])

    # Build a list of module edges, append a Scatter3D per module
    vertices   = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1]]).T
    edge_index = np.array([[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6],
                           [1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7]]).T

    n = arr[0]*arr[1]
    boxes = []
    for i in range(n):
        module_vertices = lowers[i] + vertices*size
        edges = []
        for k, l in edge_index:
            edges.append([module_vertices[k], module_vertices[l], [None,None,None]])
        edges = np.concatenate(edges)
        boxes.append(go.Scatter3d(x = edges[:,0], y = edges[:,1], z = edges[:,2],
                                  mode = 'lines',
                                  name = f'Module {i}',
                                  line = {'color': '#53EB83'}))

    return boxes


def get_truth_plots(truth):
    graph_data = []
    
    # Add points from true electronShowers
    '''
    colors = plotly.colors.qualitative.Light24
    for i, s in enumerate(output['electronShowers']):
        if len(s.voxels)<1:
            continue
        color = colors[i % (len(colors))]
        graph_data += scatter_points(np.asarray(s.voxels),markersize=2,color=color)
        graph_data[-1].name = 'True electron shower %d (edep: %.2f)' %(i,s.edep_tot)

    if len(output['electronShowers'])>0:
        #points = np.array([s.start[0:3] for s in output['electronShowers']])
        #graph_data += scatter_points(points, markersize=3, color='deepskyblue')
        #graph_data[-1].name = 'True electron shower starts'

        points = np.array([s.first_step[0:3] for s in output['electronShowers']])
        graph_data += scatter_points(points, markersize=4, color='deepskyblue')
        graph_data[-1].name = 'True electron shower 1st steps'

        #points = np.array([s.first_edep[0:3] for s in output['electronShowers']])
        #graph_data += scatter_points(points, markersize=5, color='deepskyblue')
        #graph_data[-1].name = 'True electron shower 1st edeps'
    '''

    # Add points from true photonShowers
    '''
    colors = plotly.colors.qualitative.Light24
    for i, s in enumerate(output['photonShowers']):
        if len(s.voxels)<1:
            continue
        color = colors[(i+6) % (len(colors))]
        graph_data += scatter_points(np.asarray(s.voxels),markersize=2,color=color)
        graph_data[-1].name = 'True photon shower %d (edep: %.2f)' %(i,s.edep_tot)

    if len(output['photonShowers'])>0:
        #points = np.array([s.start[0:3] for s in output['photonShowers']])
        #graph_data += scatter_points(points, markersize=3, color='darkturquoise')
        #graph_data[-1].name = 'True photon shower starts'

        points = np.array([s.first_step[0:3] for s in output['photonShowers']])
        graph_data += scatter_points(points, markersize=4, color='darkturquoise')
        graph_data[-1].name = 'True photon shower 1st steps'

        #points = np.array([s.first_edep[0:3] for s in output['photonShowers']])
        #graph_data += scatter_points(points, markersize=5, color='darkturquoise')
        #graph_data[-1].name = 'True photon shower 1st edeps'
    '''

    # Add points from true comptonShowers
    '''
    colors = plotly.colors.qualitative.Light24
    for i, s in enumerate(output['comptonShowers']):
        if len(s.voxels)<1:
            continue
        color = colors[(i+12) % (len(colors))]
        graph_data += scatter_points(np.asarray(s.voxels),markersize=2,color=color)
        graph_data[-1].name = 'True compton shower %d (edep: %.2f)' %(i,s.edep_tot)

    if len(output['comptonShowers'])>0:
        #points = np.array([s.start[0:3] for s in output['comptonShowers']])
        #graph_data += scatter_points(points, markersize=3, color='darkcyan')
        #graph_data[-1].name = 'True compton shower starts'

        points = np.array([s.first_step[0:3] for s in output['comptonShowers']])
        graph_data += scatter_points(points, markersize=4, color='darkcyan')
        graph_data[-1].name = 'True compton shower 1st steps'

        #points = np.array([s.first_edep[0:3] for s in output['comptonShowers']])
        #graph_data += scatter_points(points, markersize=5, color='darkcyan')
        #graph_data[-1].name = 'True compton shower 1st edeps'
    '''

    # Add true photons 1st steps
    # ------------------------------------
    '''
    if len(truth['gamma_first_step'])>0:
        true_gammas_first_steps = truth['gamma_first_step']
        #print(' true_gammas_first_steps: ', true_gammas_first_steps)
        graph_data += scatter_points(np.asarray(true_gammas_first_steps), markersize=5, color='magenta')
        graph_data[-1].name = 'True photons 1st steps'
    '''

    # Add compton electrons 1st steps
    # ------------------------------------
    '''
    if len(truth['compton_electron_first_step'])>0:
        compton_electrons_first_steps = truth['compton_electron_first_step']
        #print(' compton_electrons_first_steps: ', compton_electrons_first_steps)
        graph_data += scatter_points(np.asarray(compton_electrons_first_steps), markersize=5, color='green')
        graph_data[-1].name = 'True compton electrons 1st steps'
    '''
    # Add shower's true 1st (in time) step
    # ------------------------------------
    #'''
    if len(truth['shower_first_edep'])>0:
        shower_first_edep = truth['shower_first_edep']
        #print(' shower_first_edep: ', shower_first_edep)
        graph_data += scatter_points(np.asarray(shower_first_edep), markersize=5, color='red')
        graph_data[-1].name = 'True showers 1st edep'
    #'''

    # Add manually defined 3D points
    # ------------------------------------
    '''
    point_01 = np.array([469.86045002, 231.30654507, 514.07204156])
    #point_02 = np.array([406.88129432, 233.21140603, 107.01647391])
    #points = [np.asarray(point_01), np.asarray(point_02)] #,[325.2, 584.6, 312.3]]
    points = [np.asarray(point_01)]
    graph_data += scatter_points(np.asarray(points),markersize=4, color='orange')
    graph_data[-1].name = 'Manually defined point'

    point_02 = np.array([471.35858971, 244.42353517, 516.28956703])
    points = [np.asarray(point_02)]
    graph_data += scatter_points(np.asarray(points),markersize=4, color='lightgreen')
    graph_data[-1].name = 'Vtx candidate'
    '''

    # Add true pi0 decay points
    # ------------------------------------
    #'''
    if len(truth['gamma_pos'])>0:
        true_pi0_decays = truth['gamma_pos']
        graph_data += scatter_points(np.asarray(true_pi0_decays),markersize=5, color='green')
        graph_data[-1].name = 'Pi0 decay vertex (true)'
    #'''

    # Add true photon's directions (based on true pi0 decay vertex and true photon's 1st (in time) edep)
    # ------------------------------------
    '''
    if 'gamma_pos' in truth and 'shower_first_edep' in truth:
        for i, true_dir in enumerate(truth['gamma_pos']):
            vertex = truth['gamma_pos'][i]
            first_edeps = truth['shower_first_edep'][i]
            points = [vertex, first_edeps]
            graph_data += scatter_points(np.array(points),markersize=4,color='green')
            graph_data[-1].name = 'True photon %i: vertex to first edep' % i
            graph_data[-1].mode = 'lines,markers'
    '''

    # Add true photon's directions (based on true pi0 decay vertex and true photon's 1st steps)
    # ------------------------------------
    #'''
    if 'gamma_pos' in truth and 'gamma_first_step' in truth:
        for i, true_dir in enumerate(truth['gamma_pos']):
            vertex = truth['gamma_pos'][i]
            first_steps = truth['gamma_first_step'][i]
            points = [vertex, first_steps]
            graph_data += scatter_points(np.array(points),markersize=4,color='blue')
            graph_data[-1].name = 'True photon %i vtx to 1st step (einit: %.2f, edep: %.2f)'\
                                   %(i,truth['gamma_ekin'][i],truth['gamma_edep'][i])
            graph_data[-1].mode = 'lines,markers'
    #'''
    return graph_data


def draw_event(output, truth=None, draw_modules=False, **kwargs):
    '''
    Draws the output of the Pi0 reconstruction chain on a single overlayed interactive plot
    '''
    # Store all plots in a single array
    graph_data = []

    # Draw the raw input to the reconstruction chain (charge)
    charge = output['charge']
    graph_data += scatter_points(charge, markersize=2, color=np.log(charge[:,-1]), colorscale='Inferno')
    graph_data[-1].name = 'Input charge (log)'

    # Draw the semantic segmentation predictions
    segment = output['segment']
    colorscale = high_contrast_colorscale(5)
    graph_data += scatter_points(segment, markersize=2, color=segment[:,-1], colorscale=colorscale)
    graph_data[-1].name = 'Semantics'

    # Draw the reconstructed energy in each voxel
    #energy = output['energy']
    #graph_data += scatter_points(energy, markersize=2, color=np.log(energy[:,-1]), colorscale='Inferno')
    #graph_data[-1].name = 'Energy (log)'

    # Draw the proposed PPN track points, if present:
    if 'ppn_track_points' in output:
        points = output['ppn_track_points']
        graph_data += scatter_points(points,markersize=4,color='magenta')
        graph_data[-1].name = 'PPN track points'
        
    if 'ppn_shower_points' in output:
        points = output['ppn_shower_points']
        graph_data += scatter_points(points,markersize=4,color='purple')
        graph_data[-1].name = 'PPN shower points'

    # Draw the shower fragments
    if 'shower_fragments' in output:
        fragments = output['shower_fragments']
        fragment_labels = -np.ones(len(output['charge']))
        for i, f in enumerate(fragments): fragment_labels[f] = i
        colorscale = high_contrast_colorscale(len(fragments), True)
        graph_data += scatter_points(energy, markersize=2, color=fragment_labels, colorscale=colorscale)
        graph_data[-1].name = 'Shower fragments'

    # Draw the reconstructed shower objects and their reconstructed characteristics
    if 'showers' in output:
        mask, labels, hovertext, starts, dirs = [], [], [], [], []
        for i, s in enumerate(output['showers']):
            mask.extend(s.voxels)
            labels.extend(i*np.ones(len(s.voxels)))
            hovertext.extend(len(s.voxels)*[f'Shower {i}\
                <br>Start: ({s.start[0]:0.1f},{s.start[1]:0.1f},{s.start[2]:0.1f})\
                <br>Direction: ({s.direction[0]:0.2f},{s.direction[1]:0.2f},{s.direction[2]:0.2f})\
                <br>Energy: {s.energy:0.1f} MeV'])
            starts.append(s.start)
            dirs.append(s.direction)

        # Draw the shower voxels with their legend
        n_showers = len(output['showers'])
        colorscale = high_contrast_colorscale(n_showers)
        graph_data += scatter_points(energy[mask], markersize=2, color=labels, hovertext=hovertext, colorscale=colorscale)
        graph_data[-1].name = 'Showers'

        # Draw the shower starts
        graph_data += scatter_points(np.vstack(starts), markersize=4, color='yellow')
        graph_data[-1].name = 'Shower starts (reco)'

        # Draw the shower directions
        dir_segments = np.concatenate([[starts[i], starts[i]+10*dirs[i], [None,None,None]] for i in range(n_showers)])
        graph_data += scatter_points(dir_segments, color='black')
        graph_data[-1].name = 'Shower directions'
        graph_data[-1].mode = 'lines'
        graph_data[-1]['line']['width'] = 5

    # Draw the reconstructed Pi0 if any are found in the event
    if 'matches' in output and len(output['matches']):

        # Draw the
        for i, match in enumerate(output['matches']):
            v = output['vertices'][i]
            idx1, idx2 = match
            s1, s2 = output['showers'][idx1].start, output['showers'][idx2].start
            points = [v, s1, v, s2]
            graph_data += scatter_points(np.array(points),color='red')
            graph_data[-1].name = 'Reconstructed pi0<br>(%.2f MeV)' % output['masses'][i]
            graph_data[-1].mode = 'lines,markers'

        # Draw the decay vertices
        graph_data += scatter_points(np.vstack(output['vertices']), markersize=4, color='lightgreen')
        graph_data[-1].name = 'Pi0 decay vertex (reco)'

    # Add outer module dimensions (TODO: Check dimensions, probably add active volumes instead of outer module edges)
    if draw_modules:
        graph_data += module_boxes()
    
    # Add truth information
    if truth != None:
        graph_data += get_truth_plots(truth)

    # Draw
    iplot(go.Figure(data=graph_data,layout=layout(**kwargs)))
