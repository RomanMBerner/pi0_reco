import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot
from mlreco.visualization.points import scatter_points

def layout(width=1024, height=768, xrange=(0,768), yrange=(0,768), zrange=(0,768), dark=False, aspectmode='cube'):
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

def draw_event(output, truth, **kwargs):

    graph_data = []

    # Draw voxels with cluster labels
    # ------------------------------------
    energy = output['energy']
    graph_data += scatter_points(energy,markersize=2,color=energy[:,-1],colorscale='Inferno')
    graph_data[-1].name = 'Energy'


    # Add points from true electronShowers
    # ------------------------------------
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
    # ------------------------------------
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
    # ------------------------------------
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

    # Add points from recoShowers
    # ------------------------------------
    #'''
    colors = plotly.colors.qualitative.Light24
    for i, s in enumerate(output['showers']):
        if len(s.voxels)<1:
            continue
        color = colors[(i+18) % (len(colors))]
        points = energy[s.voxels]
        graph_data += scatter_points(points,markersize=2,color=color)
        #graph_data[-1].name = 'Reco shower %d (n_edeps=%d, edep=%.2f, L_e=%.3f, L_p=%.3f, dir=[%.2f,%.2f,%.2f])' % (i,len(s.voxels),s.energy,s.L_e,s.L_p,s.direction[0],s.direction[1],s.direction[2])
        graph_data[-1].name = 'Reco shower %d (edep: %.2f, dir: %.1f %.1f %.1f)' %(i,s.energy,s.direction[0],s.direction[1],s.direction[2])

    if len(output['showers'])>0:
        points = np.array([s.start for s in output['showers']])
        graph_data += scatter_points(points, markersize=3, color='gold')
        graph_data[-1].name = 'Reco shower starts'

        points = np.array([s.start for s in output['showers']])
        dirs = np.array([s.direction for s in output['showers']])
        cone_start = points[:,:3]
        arrows = go.Cone(x=cone_start[:,0], y=cone_start[:,1], z=cone_start[:,2],
                         u=-dirs[:,0], v=-dirs[:,1], w=-dirs[:,2],
                         sizemode='absolute', sizeref=1.0, anchor='tip',
                         showscale=False, opacity=0.4)
        #graph_data.append(arrows)
    #'''


    # Add true pi0 decay points
    # ------------------------------------
    #'''
    if len(truth['gamma_pos'])>0:
        true_pi0_decays = truth['gamma_pos']
        graph_data += scatter_points(np.asarray(true_pi0_decays),markersize=5, color='green')
        graph_data[-1].name = 'True pi0 decay vertices'
    #'''

    # Add reconstructed pi0 decay points
    # ------------------------------------
    #'''
    try:
        reco_pi0_decays = output['vertices']
        graph_data += scatter_points(np.asarray(reco_pi0_decays),markersize=4, color='lightgreen')
        graph_data[-1].name = 'Reconstructed pi0 decay vertices'
    except:
        pass
    #'''

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
    '''
    if len(truth['shower_first_edep'])>0:
        shower_first_edep = truth['shower_first_edep']
        #print(' shower_first_edep: ', shower_first_edep)
        graph_data += scatter_points(np.asarray(shower_first_edep), markersize=5, color='red')
        graph_data[-1].name = 'True showers 1st steps'
    '''

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

    # Add points predicted by PPN
    # ------------------------------------
    #'''
    if 'ppn_track_points' in output:
        points = np.array([i.ppns for i in output['ppn_track_points']])
        graph_data += scatter_points(points,markersize=4,color='magenta')
        graph_data[-1].name = 'PPN track points'
    # if 'ppn_shower_points' in output:
    #     points = np.array([i.ppns for i in output['ppn_shower_points']])
    #     graph_data += scatter_points(points,markersize=4,color='purple')
    #     graph_data[-1].name = 'PPN shower points'
    '''
    if output['PPN_michel_points']:
        points = np.array([i.ppns for i in output['PPN_michel_points']])
        graph_data += scatter_points(points,markersize=4,color='purple')
        graph_data[-1].name = 'PPN michel points'
    if output['PPN_delta_points']:
        points = np.array([i.ppns for i in output['PPN_delta_points']])
        graph_data += scatter_points(points,markersize=4,color='purple')
        graph_data[-1].name = 'PPN delta points'
    if output['PPN_LEScat_points']:
        points = np.array([i.ppns for i in output['PPN_LEScat_points']])
        graph_data += scatter_points(points,markersize=4,color='purple')
        graph_data[-1].name = 'PPN LEScatter points'
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
            graph_data[-1].name = 'True photon %i vertex to first step (einit: %.2f, edep: %.2f)'\
                                   %(i,truth['gamma_ekin'][i],truth['gamma_edep'][i])
            graph_data[-1].mode = 'lines,markers'
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

    # Add reconstructed pi0 decays, join vertex to start points
    # ------------------------------------
    #'''
    if 'matches' in output:
        for i, match in enumerate(output['matches']):
            v = output['vertices'][i]
            idx1, idx2 = match
            s1, s2 = output['showers'][idx1].start, output['showers'][idx2].start
            points = [v, s1, v, s2]
            graph_data += scatter_points(np.array(points),color='red')
            graph_data[-1].name = 'Reconstructed pi0 (%.2f MeV)' % output['masses'][i]
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
    iplot(go.Figure(data=graph_data,layout=layout(**kwargs)))
