def matrix_dispersion(Nx, Ny, t, kx, ky):
  A = initCubicTBH(Nx, Ny, 1, 0, t, cyclic=False)
  nsites = Nx*Ny
  B = numpy.zeros((nsites,nsites))
  for l in range(nsites):
    i = l % Nx
    j = int(l / Nx)
    if i+1==Nx: 
      lnn = j*Nx
      B[l,lnn]+=t*exp(1j*kx)
      B[lnn,l]+=t*exp(-1j*kx)
    if j+1==Ny: 
      lnn = i
      B[l,lnn]+=t*exp(1j*ky)
      B[lnn,l]+=t*exp(-1j*ky) 
  return A+B

def fill_H0_k(H0_k,Nx,Ny,t,ks, mus):
  nsites = Nx*Ny 
  for kxi,kx in enumerate(ks):
    for kyi,ky in enumerate(ks):
      epsk = matrix_dispersion(Nx,Ny,t,kx,ky)
      H0_k[kxi,kyi,:nsites,:nsites] = epsk - numpy.diag(mus) 
      H0_k[kxi,kyi,nsites:,nsites:] = numpy.diag(mus) - epsk

def get_G(G_IaJb_k_iw, H0_k, Sigma_IaJb_imp_iw):
  iws = [iw.value for iw in Sigma_IaJb_imp_iw.mesh]
  dummy1, nk, nk, nI, nI = numpy.shape(G_IaJb_k_iw)
  for iwi, iw in enumerate(iws):
    for kxi in range(nk):
      for kyi in range(nk):
        G_IaJb_k_iw[iwi,kxi,kyi,:,:] = numpy.linalg.inv(iw*numpy.eye(N)-H0_k[kxi,kyi,:,:]-Sigma_IaJb_imp_iw.data[iwi,:,:])

def get_G_loc(G_IaJb_loc_tau, G_IaJb_loc_iw, G_IaJb_k_iw):
  G_IaJb_loc_iw.data[:,:,:] = numpy.sum(G_IaJb_k_iw, axes=(1,2))
  fit_fermionic_gf_tail(G_IaJb_loc_iw, starting_iw=14.0, no_loc=False, overwrite_tail=True, max_order=5)
  G_IaJb_loc_tau << InverseFourier(G_IaJb_loc_iw)

def get_n(G_IaJb_loc_tau):
  ntau, nIa, nJb = numpy.shape(G_IaJb_loc_tau.data)
  tot = 0
  nsites = nIa/2
  for Ia in range(nIa):
    if Ia>=nsites:
      tot+= -G_IaJb_loc_tau.data[0,Ia,Ia]
    else:
      tot+= -G_IaJb_loc_tau.data[-1,Ia,Ia]
  return tot/nIa

def get_Gweiss(Gweiss_IaJb_iw, G_IaJb_loc_iw, Sigma_IaJb_imp_iw):
  Gweiss_IaJb_iw << inverse(inverse(G_IaJb_loc_iw)+Sigma_IaJb_imp_iw)
  fit_fermionic_gf_tail(Gweiss_IaJb_iw, starting_iw=14.0, no_loc=False, overwrite_tail=True, max_order=5)

def get_Sigma(solver, Sigma_imp_iw, Gweiss_iw, Us, nambu=False, solver_data_package=None, su2=True, nambu=False):
  if nambu:
    solver.G0_iw['nambu'] << Gweiss_iw
  else: assert False, "not implemented..."

  solvers.ctint.run(
    solver=solver, 
    Us=Us,      
    nambu=nambu,
    alpha=0.5,
    delta=0.1,
    n_cycles=20000,
    max_time = 5*60,
    solver_data_package,
    only_sign = False
  )
  if su2:
    if nambu:  
      symm_Sig = solver.Sigma_iw['nambu'].copy()
      niw, Nstates, Nstates = numpy.shape(symm_Sig.data)
      nsites = Nstates/2        
      for i in range(nsites):
        for j in range(nsites):
          temp = ( symm_Sig.data[:,i,j]-numpy.conj(symm_Sig.data[:,i+nsites,j+nsites]) )/2.0
          Sigma_imp_iw.data[:,i,j] = temp
          Sigma_imp_iw.data[:,i+nsites,j+nsites] = -numpy.conj(temp)
      for i in range(nsites):
        for j in range(nsites):
          temp = ( symm_Sig.data[:,i+nsites,j]+numpy.conj(symm_Sig.data[:,i,j+nsites]) )/2.0
          Sigma_imp_iw.data[:,i+nsites,j] = temp
          Sigma_imp_iw.data[:,i,j+nsites] = numpy.conj(temp)
    else:
      Sigma_imp_iw << (solver.Sigma_iw['up']+solver.Sigma_iw['dn'])/2.0    
  else: 
    if nambu:  
      Sigma_imp_iw << solver.Sigma_iw['nambu']
    else: 
      Sigma_imp_iw << solver.Sigma_iw

def cellular_data( nk, niw, ntau, Nx,Ny, beta ):
  nsites = Nx*Ny
  dt = data() 
  dt.nk = nk
  dt.ks = numpy.linspace(0.0,2.0*numpy.pi,nk,endpoint=False)
  dt.niw = niw
  dt.ntau = ntau
  dt.nsites = nsites
  dt.Nx = Nx
  dt.Ny = Ny
  dt.beta = beta
  dt.T = 1.0/beta
  dt.blocks = blocks 
  print "Adding lattice Green's function..."  
  AddNumpyData(dt, ['G_IaJb_k_iw'], (2*niw, nk,nk, 2*nsites, 2*nsites) ) #optimizing further, we don't even need G0_ij_iw
  print "Adding loc and imp Green's functions..."  
  for Q in ['G_IaJb_loc_iw','Sigma_IaJb_imp_iw','Gweiss_IaJb_iw']
    vars(dt)[Q] = GfImFreq(indices = range(2*nsites), beta = beta, n_points = niw, statistic = 'Fermion')
  for Q in ['G_IaJb_loc_tau']
    vars(dt)[Q] = GfImTime(indices = range(2*nsites), beta = beta, n_points = niw, statistic = 'Fermion')

  dt.iws = numpy.array([iw.value for iw in dt.G_IaJb_loc_iw.mesh]) 
  print "Adding Hamiltonian and parameters..."  
  AddNumpyData(dt, ['H0_k'], (nk,nk, 2*nsites,2*nsites))
  AddNumpyData(dt, ['mus','Us'], (nsites))
  print "Done preparing containers"  
  return dt

def cellular_set_calc( dt, solver_data_package=None, nambu=True, su2=True ):
  dt.get_H0_k = lambda: fill_H0_k(dt.H0_k,dt.Nx,dt.Ny,dt.t,dt.ks, dt.mus)

  dt.get_Gweiss = lambda: get_Gweiss(dt.Gweiss_IaJb_iw, dt.G_IaJb_loc_iw, dt.Sigma_IaJb_imp_iw)

  dt.get_Sigma = lambda: get_Sigma(dt.solver, dt.Sigma_IaJb_imp_iw, dt.Gweiss_IaJb_iw, dt.Us, dt.nambu, solver_data_package, su2, nambu)

  dt.get_G = lambda: get_G(dt.G_IaJb_k_iw, dt.H0_k, dt.Sigma_IaJb_imp_iw)

  dt.get_G_loc = lambda: get_G_loc(dt.G_IaJb_loc_iw, dt.G_IaJb_k_iw)

  dt.get_n = lambda: [dt.get_G_loc(), get_n(dt.G_IaJb_loc_tau)][-1]

