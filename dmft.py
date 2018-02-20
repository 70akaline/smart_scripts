import data
from data import *
from data import data

def dmft_data( niw, ntau, nk, beta, blocks = ['up'] ):
  dt = data() 
  dt.blocks = blocks 
  AddGfData(dt, ['G_imp_iw', 'G_loc_iw', 'Sigma_imp_iw', 'Gweiss_iw'], blocks, 1, niw, beta, domain = 'iw', suffix='', statistic='Fermion')
  AddGfData(dt, ['Sigma_imp_tau',' Gweiss_tau'], blocks, 1, ntau, beta, domain = 'tau', suffix='', statistic='Fermion')
  dt.ks = numpy.linspace(0,2.0*numpy.pi, nk, endpoint=False)
  AddBlockNumpyData(dt, ['G0_k_iw', 'G_k_iw','G_r_iw','Sigma_k_iw','Sigma_r_iw'], blocks, (niw*2,nk,nk))
  AddBlockNumpyData(dt, ['epsilon_k'], blocks, (nk,nk))




