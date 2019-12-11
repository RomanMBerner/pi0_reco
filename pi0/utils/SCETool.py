import math
import ROOT

class SCETool(object):
  E_nominal = 0.4867
  cryoTemp = 87.3

  WF_top = 604.0
  WF_bottom = 5.2
  WF_upstream = -0.5
  WF_downstream = 695.3
  WF_anode = 360.0

  ModBoxA = 0.930 # recomb constant A used in LArG4 (DO NOT CHANGE)
  ModBoxB = 0.212 # recomb constant B used in LArG4 (DO NOT CHANGE)
  LAr_density = 1.3973 # LAr density (DO NOT CHANGE)

  def __init__(self, filepath='SCE_DataDriven_180kV_v3.root'):
    self.filepath = filepath
    self.fileHists = None

    self.RecoFwdX_Neg = None
    self.RecoFwdY_Neg = None
    self.RecoFwdZ_Neg = None
    self.RecoFwdX_Pos = None
    self.RecoFwdY_Pos = None
    self.RecoFwdZ_Pos = None

    self.RecoBkwdX_Neg = None
    self.RecoBkwdY_Neg = None
    self.RecoBkwdZ_Neg = None
    self.RecoBkwdX_Pos = None
    self.RecoBkwdY_Pos = None
    self.RecoBkwdZ_Pos = None

    self.RecoEFieldX_Neg = None
    self.RecoEFieldY_Neg = None
    self.RecoEFieldZ_Neg = None
    self.RecoEFieldX_Pos = None
    self.RecoEFieldY_Pos = None
    self.RecoEFieldZ_Pos = None

  def LoadHists(self):
    '''
    Loads histograms from file

    '''
    self.fileHists = ROOT.TFile(self.filepath)

    self.RecoFwdX_Neg = self.fileHists.Get('RecoFwd_Displacement_X_Neg')
    self.RecoFwdY_Neg = self.fileHists.Get('RecoFwd_Displacement_Y_Neg')
    self.RecoFwdZ_Neg = self.fileHists.Get('RecoFwd_Displacement_Z_Neg')
    self.RecoFwdX_Pos = self.fileHists.Get('RecoFwd_Displacement_X_Pos')
    self.RecoFwdY_Pos = self.fileHists.Get('RecoFwd_Displacement_Y_Pos')
    self.RecoFwdZ_Pos = self.fileHists.Get('RecoFwd_Displacement_Z_Pos')

    self.RecoBkwdX_Neg = self.fileHists.Get('RecoBkwd_Displacement_X_Neg')
    self.RecoBkwdY_Neg = self.fileHists.Get('RecoBkwd_Displacement_Y_Neg')
    self.RecoBkwdZ_Neg = self.fileHists.Get('RecoBkwd_Displacement_Z_Neg')
    self.RecoBkwdX_Pos = self.fileHists.Get('RecoBkwd_Displacement_X_Pos')
    self.RecoBkwdY_Pos = self.fileHists.Get('RecoBkwd_Displacement_Y_Pos')
    self.RecoBkwdZ_Pos = self.fileHists.Get('RecoBkwd_Displacement_Z_Pos')

    self.RecoEFieldX_Neg = self.fileHists.Get('Reco_ElecField_X_Neg')
    self.RecoEFieldY_Neg = self.fileHists.Get('Reco_ElecField_Y_Neg')
    self.RecoEFieldZ_Neg = self.fileHists.Get('Reco_ElecField_Z_Neg')
    self.RecoEFieldX_Pos = self.fileHists.Get('Reco_ElecField_X_Pos')
    self.RecoEFieldY_Pos = self.fileHists.Get('Reco_ElecField_Y_Pos')
    self.RecoEFieldZ_Pos = self.fileHists.Get('Reco_ElecField_Z_Pos')

  def GetFwdOffset(self, x_true, y_true, z_true, comp, whichSide):
    '''
    Calculates the 'forward offset' at true postion (``x_true``,
    ``y_true``,``z_true``). The forward offset is observed shift in position of
    charge deposited at a given x,y,z due to the space charge effect.

    :param x_true: true x

    :param y_true: true y

    :param z_true: true z

    :param comp: offset component to return (1=x, 2=y, 3=z)

    :param whichSide: >0 if spacepoint is reconstructed on beam left anode, <0 if reconstructed on beam right anode

    :returns: specified forward offset

    '''
    if whichSide > 0:
      if x_true > self.WF_anode - 0.001:
        x_true = self.WF_anode - 0.001
      elif x_true < 0.001:
        x_true = 0.001
    else:
      if(x_true < -1.0 * self.WF_anode + 0.001):
        x_true = -1.0 * self.WF_anode + 0.001
      elif x_true > -0.001:
        x_true = -0.001

    if y_true < self.WF_bottom + 0.001:
      y_true = self.WF_bottom + 0.001
    elif y_true > self.WF_top - 0.001:
      y_true = self.WF_top - 0.001

    if z_true < self.WF_upstream + 0.001:
      z_true = self.WF_upstream + 0.001
    elif z_true > self.WF_downstream - 0.001:
      z_true = self.WF_downstream-0.001

    offset = 0.0
    if whichSide > 0:
      if comp == 1:
        offset = self.RecoFwdX_Pos.Interpolate(x_true,y_true,z_true)
      elif comp == 2:
        offset = self.RecoFwdY_Pos.Interpolate(x_true,y_true,z_true)
      elif comp == 3:
        offset = self.RecoFwdZ_Pos.Interpolate(x_true,y_true,z_true)
    else:
      if comp == 1:
        offset = self.RecoFwdX_Neg.Interpolate(x_true,y_true,z_true)
      elif(comp == 2):
        offset = self.RecoFwdY_Neg.Interpolate(x_true,y_true,z_true)
      elif comp == 3:
        offset = self.RecoFwdZ_Neg.Interpolate(x_true,y_true,z_true)

    return offset

  def GetBkwdOffset(self, x_reco, y_reco, z_reco, comp, whichSide):
    '''
    Calculates the 'backward offset' at reconstructed postion (``x_reco``,
    ``y_reco``,``z_reco``). The backward offset is the shift in x,y,z required to
    move a spacepoint reconstructed at the anode to its true x,y,z within the
    tpc volume

    :param x_reco: reconstructed x

    :param y_reco: reconstructed y

    :param z_reco: reconstructed z

    :param comp: offset component to return (1=x, 2=y, 3=z)

    :param whichSide: >0 if spacepoint is reconstructed on beam left anode, <0 if reconstructed on beam right anode

    :returns: specified backward offset

    '''
    if whichSide > 0:
      if x_reco > self.WF_anode - 0.001:
        x_reco = self.WF_anode - 0.001
      elif x_reco < 0.001:
        x_reco = 0.001
    else:
      if x_reco < -1.0 * self.WF_anode + 0.001:
        x_reco = -1.0 * self.WF_anode + 0.001;
      elif x_reco > -0.001:
        x_reco = -0.001

    if y_reco < self.WF_bottom + 0.001:
      y_reco = self.WF_bottom + 0.001
    elif y_reco > self.WF_top - 0.001:
      y_reco = self.WF_top - 0.001

    if z_reco < self.WF_upstream + 0.001:
      z_reco = self.WF_upstream + 0.001
    elif z_reco > self.WF_downstream - 0.001:
      z_reco = self.WF_downstream - 0.001

    offset = 0.0
    if whichSide > 0:
      if comp == 1:
        offset = self.RecoBkwdX_Pos.Interpolate(x_reco,y_reco,z_reco)
      elif comp == 2:
        offset = self.RecoBkwdY_Pos.Interpolate(x_reco,y_reco,z_reco)
      elif comp == 3:
        offset = self.RecoBkwdZ_Pos.Interpolate(x_reco,y_reco,z_reco)
    else:
      if comp == 1:
        offset = self.RecoBkwdX_Neg.Interpolate(x_reco,y_reco,z_reco)
      elif comp == 2:
        offset = self.RecoBkwdY_Neg.Interpolate(x_reco,y_reco,z_reco)
      elif(comp == 3):
        offset = self.RecoBkwdZ_Neg.Interpolate(x_reco,y_reco,z_reco)

    return offset

  def GetLocalEField(self, x_true, y_true, z_true, whichSide):
    '''
    Get the magnitude of the local electric field based on the data-driven space
    charge model

    :param x_true: true x

    :param y_true: true y

    :param z_true: true z

    :param whichSide: >0 if spacepoint is reconstructed on beam left anode, <0 if reconstructed on beam right anode

    :returns: electric field magnitude at specified point

    '''
    if whichSide > 0:
      if x_true > self.WF_anode - 0.001:
        x_true = self.WF_anode - 0.001
      elif x_true < 0.001:
        x_true = 0.001;
    else:
      if x_true < -1.0*self.WF_anode + 0.001:
        x_true = -1.0*self.WF_anode + 0.001
      elif x_true > -0.001:
        x_true = -0.001

    if y_true < self.WF_bottom + 0.001:
      y_true = self.WF_bottom + 0.001
    elif y_true > self.WF_top - 0.001:
      y_true = self.WF_top - 0.001

    if z_true < self.WF_upstream + 0.001:
      z_true = self.WF_upstream + 0.001
    elif z_true > self.WF_downstream - 0.001:
      z_true = self.WF_downstream - 0.001

    Emag = self.E_nominal
    if whichSide > 0:
      Emag = self.E_nominal*(1.0 + math.sqrt(ROOT.TMath.Power(1.0 + self.RecoEFieldX_Pos.Interpolate(x_true,y_true,z_true),2) + ROOT.TMath.Power(self.RecoEFieldY_Pos.Interpolate(x_true,y_true,z_true),2) + ROOT.TMath.Power(self.RecoEFieldZ_Pos.Interpolate(x_true,y_true,z_true),2)) - 1.0)
    else:
      Emag = self.E_nominal*(1.0 + math.sqrt(ROOT.TMath.Power(1.0 + self.RecoEFieldX_Neg.Interpolate(x_true,y_true,z_true),2) + ROOT.TMath.Power(self.RecoEFieldY_Neg.Interpolate(x_true,y_true,z_true),2) + ROOT.TMath.Power(self.RecoEFieldZ_Neg.Interpolate(x_true,y_true,z_true),2)) - 1.0)

    return Emag

def main():
  import matplotlib.pyplot as plt
  import numpy as np

  tool = SCETool()
  tool.LoadHists()

  x_pos = np.linspace(-tool.WF_anode, tool.WF_anode, 100)
  y_pos = np.linspace(tool.WF_bottom, tool.WF_top, 100)
  z = (tool.WF_downstream - tool.WF_upstream)/2
  side_pos = np.where(np.array(x_pos) > 0, 1, -1)
  dX_fwd = [[tool.GetFwdOffset(x, y, z, 1, side) for x,side in zip(x_pos, side_pos)] for y in y_pos]
  dY_fwd = [[tool.GetFwdOffset(x, y, z, 2, side) for x,side in zip(x_pos, side_pos)] for y in y_pos]
  offset_fwd = [[math.sqrt(tool.GetFwdOffset(x, y, z, 1, side)**2+tool.GetFwdOffset(x, y, z, 2, side)**2+tool.GetFwdOffset(x, y, z, 3, side)**2) for x,side in zip(x_pos, side_pos)] for y in y_pos]
  dX_bkwd = [[tool.GetBkwdOffset(x, y, z, 1, side) for x,side in zip(x_pos, side_pos)] for y in y_pos]
  dY_bkwd = [[tool.GetBkwdOffset(x, y, z, 2, side) for x,side in zip(x_pos, side_pos)] for y in y_pos]
  e_field = [[tool.GetLocalEField(x, y, z, side) for x,side in zip(x_pos, side_pos)] for y in y_pos]
  def recombination(field):
    dEdx = 2.1 # MIP dE/dx (in MeV/cm)
    xi = (tool.ModBoxB * dEdx) / (tool.LAr_density * field)
    return math.log(tool.ModBoxA + xi)/xi
  recomb = [[recombination(field) for field in field_row] for field_row in e_field]

  def make_div_plot(name, x, y, z, title, xlabel, ylabel, zlabel, central_val=0,
    min_val=None, max_val=None, tick_fmt='%.1f'):
    '''
    Creates a filled-contour plot with a color scale diverging from the
    ``central_val``

    '''
    plt.figure(name)
    plt.clf()
    scale = max(abs(central_val-abs(np.min(z))),abs(central_val-abs(np.max(z))))
    if not min_val:
      min_val = central_val-scale
    if not max_val:
      max_val = central_val+scale
    plt.contourf(x, y, z, 256, cmap='coolwarm', vmin=min_val, vmax=max_val)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ticks = np.linspace(np.min(z),np.max(z),10)
    cb = plt.colorbar(ticks=ticks)
    cb.ax.set_yticklabels([tick_fmt % val for val in ticks])
    cb.set_label(zlabel, rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig(name+'.png')

  make_div_plot('dX_YvsX_Fwd', x_pos, y_pos, dX_fwd,
    'Sim. $\Delta X$ @ $Z_{true}$ = %.1fcm' % z, '$X_{true}$ [cm]',
    '$Y_{true}$ [cm]', '$\Delta X$ [cm]')
  make_div_plot('dY_YvsX_Fwd', x_pos, y_pos, dY_fwd,
    'Sim. $\Delta Y$ @ $Z_{true}$ = %.1fcm' % z, '$X_{true}$ [cm]',
    '$Y_{true}$ [cm]', '$\Delta Y$ [cm]')

  make_div_plot('dX_YvsX_Bkwd', x_pos, y_pos, dX_bkwd,
    'Sim. $\Delta X$ @ $Z_{reco}$ = %.1fcm' % z, '$X_{reco}$ [cm]',
    '$Y_{reco}$ [cm]', '$\Delta X$ [cm]')
  make_div_plot('dY_YvsX_Bkwd', x_pos, y_pos, dY_bkwd,
    'Sim. $\Delta Y$ @ $Z_{reco}$ = %.1fcm' % z, '$X_{reco}$ [cm]',
    '$Y_{reco}$ [cm]', '$\Delta Y$ [cm]')

  make_div_plot('Efield_YvsX', x_pos, y_pos, e_field,
    'E Field Mag. @ $Z_{true}$ = %.1fcm' % z, '$X_{true}$ [cm]',
    '$Y_{true}$ [cm]', 'E [kV/cm]', central_val=tool.E_nominal, tick_fmt='%.3f')
  make_div_plot('recomb_YvsX', x_pos, y_pos, recomb,
    'Recomb. Factor @ $Z_{true}$ = %.1fcm' % z, '$X_{true}$ [cm]',
    '$Y_{true}$ [cm]', '', central_val=recombination(tool.E_nominal), tick_fmt='%.3f')

if __name__ == '__main__':
  main()
