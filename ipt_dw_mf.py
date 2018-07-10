from smart_scripts import *

import sys
try:
  sys.path.insert(0,'/home/jaksa/parallel_dyson/build')
  from parallel_dyson import *
except:
  print "parallel_dyson module not found, this optimization not available in ipt_dw_mf!"

def ipt_dw_mf_data( niw, ntau, nk, beta ):
  dt = data() 
  dt.niw = niw
  dt.ntau = ntau
  dt.nk = nk
  dt.beta = beta
  dt.T = 1.0/beta
  dt.iws = numpy.array([1j*(2.0*n+1)*pi*dt.T for n in range(-niw,niw)])
  dt.blocks = ['up']
  print "Adding loc, imp and n.n. anomalous Green's functions..."  
  AddGfData(dt, ['G_loc_iw', 'Sigma_imp_iw','Gweiss_iw', 'F_r10_iw'], dt.blocks, 1, niw, beta, domain = 'iw', suffix='', statistic='Fermion')
  AddGfData(dt, ['G_loc_tau', 'Sigma_imp_tau','Gweiss_tau','F_r10_tau'], dt.blocks, 1, ntau, beta, domain = 'tau', suffix='', statistic='Fermion')
  print "Adding lattice Green's functions..."  
  AddNumpyData(dt, ['G_ab_k_iw'], (2*niw, nk,nk, 2,2) )
  AddNumpyData(dt, ['F_k_iw','F_r_iw'], (2*niw, nk,nk) )
  AddNumpyData(dt, ['H0'], (nk,nk,2,2))
  AddNumpyData(dt, ['ks'], (nk))
  AddScalarData(dt, ['mu','F','g','B','n','n0','mu0tilde'], vals=None)
  print "Done preparing containers"  
  return dt

def ipt_dw_mf_set_calc(
  dt, 
  epsk = lambda kx,ky, t: 2*t*(numpy.cos(kx)+numpy.cos(ky)),
  gk = lambda kx,ky, g: g*(numpy.cos(kx)-numpy.cos(ky))  
):
  def H0(kx,ky, mu, F):
    return numpy.array([
      [ epsk(kx,ky,dt.t)-mu,  F*gk(kx,ky,dt.g) ],
      [ F*gk(kx,ky,dt.g),     mu-epsk(kx,ky,dt.t) ],
    ])
 
  def fill_H0(): 
    for kxi,kx in enumerate(dt.ks):
      for kyi,ky in enumerate(dt.ks):
        dt.H0[kxi,kyi,:,:] = H0(kx,ky,dt.mu,dt.F)
    
  dt.get_H0 = lambda: fill_H0()

  def get_G():
    for iwi, iw in enumerate(dt.iws):
      sig = dt.Sigma_imp_iw['up'].data[iwi,0,0]
      nambu_sig = numpy.array([[sig,0],[0,-numpy.conj(sig)]])
      for kxi,kx in enumerate(dt.ks):
        for kyi,ky in enumerate(dt.ks):
          dt.G_ab_k_iw[iwi,kxi,kyi,:,:] = numpy.linalg.inv(iw*numpy.eye(2)-dt.H0[kxi,kyi]-nambu_sig)

  #dt.get_G = lambda: get_G()
  dt.get_G = lambda: parallel_get_Nambu_G(dt.iws, dt.H0, dt.Sigma_imp_iw['up'].data[:,0,0], dt.G_ab_k_iw)

  def get_G_loc():
    for iwi, iw in enumerate(dt.iws):
      dt.G_loc_iw['up'].data[iwi,0,0] = numpy.sum(dt.G_ab_k_iw[iwi,:,:,0,0])/(dt.nk**2)
    fit_fermionic_gf_tail(dt.G_loc_iw['up'])
    dt.G_loc_tau['up'] << InverseFourier(dt.G_loc_iw['up'])
  dt.get_G_loc = lambda: get_G_loc()
 
  dt.get_mu = lambda: dt.mu

  def set_mu(mu):
    dt.mu = mu
  dt.set_mu = lambda mu: set_mu(mu)

  dt.get_n = lambda: [
    fill_H0(),
    dt.get_G(),
    dt.get_G_loc(),
    -dt.G_loc_tau['up'].data[-1,0,0].real
  ][-1]

  def get_F():    
    dt.F_k_iw[:,:,:] = dt.G_ab_k_iw[:,:,:,0,1]
    dt.F_r_iw[:,:,:] =  [ numpy.fft.ifft2(dt.F_k_iw[iwi,:,:]) for iwi,iw in enumerate(dt.iws)]
    dt.F_r10_iw['up'].data[:,0,0] = dt.F_r_iw[:,1,0]
    dt.F_r10_tau['up'] << InverseFourier(dt.F_r10_iw['up'])
    dt.F = dt.F_r10_tau['up'].data[0,0,0].real 
  
  dt.get_F = lambda: get_F()

  dt.get_Gweiss_iw = lambda mu0tilde=0: orbital_space_dyson_get_G0(
    dt.Gweiss_iw['up'],
    dt.G_loc_iw['up'],
    dt.Sigma_imp_iw['up'],
    mu0tilde
  )

  def get_Gweiss_tau():
    impose_real_valued_in_imtime(dt.Gweiss_iw['up'])  
    fit_fermionic_gf_tail(dt.Gweiss_iw['up'])
    dt.Gweiss_tau['up'] << InverseFourier(dt.Gweiss_iw['up']) 
  dt.get_Gweiss_tau = lambda: get_Gweiss_tau()

  dt.get_mu0tilde = lambda: dt.mu0tilde
  def set_mu0tilde(mu0tilde):
    dt.mu0tilde = mu0tilde
  dt.set_mu0tilde = lambda mu0tilde: set_mu0tilde(mu0tilde)
  dt.get_n0 = lambda: [
    dt.get_Gweiss_iw(dt.mu0tilde),
    dt.get_Gweiss_tau(),
    -dt.Gweiss_tau['up'].data[-1,0,0].real
  ][2]

  dt.get_Sigma_imp_tau = lambda: get_Sigma_imp_tau_from_Gweiss_tau(dt.Sigma_imp_tau['up'], dt.Gweiss_tau['up'], dt.U)

  dt.get_B = lambda n, U, mu0tilde: ((1.0-n)*U + mu0tilde)/(n*(1.0-n)*U**2)      
  def get_Sigma_imp_iw():      
      dt.B = dt.get_B(dt.n, dt.U, dt.mu0tilde)
      dt.Sigma_imp_iw['up'] << Fourier(dt.Sigma_imp_tau['up'])
      dt.Sigma_imp_iw['up'] << dt.n*dt.U + dt.Sigma_imp_iw['up']*inverse(1.0-dt.B*dt.Sigma_imp_iw['up'])
      fit_fermionic_sigma_tail(dt.Sigma_imp_iw['up'])
  dt.get_Sigma_imp_iw = lambda: get_Sigma_imp_iw()

def ipt_dw_mf_set_params_and_initialize(
  dt,
  n, mu, g, T, U, t, fixed_n = True, ph_symmetry = True, initial_F=0.0, initial_Gweiss_iw=None,
  filename=None
):
  if filename is None: 
    filename = "ipt_dw_mf"
    if fixed_n:
      filename += ".n%.4f"%n
    else:
      filename += ".mu%.4f"%mu
    filename += ".U%.4f.T%.4f"%(U,T)
    if initial_F!=0.0: filename+=".from_sc"
    else: filename+=".from_normal"

  dt.archive_name = filename
  dt.dump = lambda dct: DumpData(dt, filename, Qs=[], exceptions=['G_ab_k_iw','F_k_iw','F_r_iw'], dictionary=dct)
  dt.dump_final = lambda dct: DumpData(dt, filename, Qs=[], exceptions=[], dictionary=dct)

  dt.ks = numpy.linspace(0,2.0*numpy.pi,dt.nk,endpoint=False)
  dt.n = n #if mu is fized, n is used as initial guess
  dt.mu = mu #if fixed_n, mu is used as initial guess
  dt.mu0tilde = -mu
  dt.U = U
  dt.g =g
  dt.t = t    
  dt.F = initial_F
  dt.fixed_n = fixed_n
  dt.ph_symmetry = ph_symmetry

  dt.Sigma_imp_iw << U*n
  if initial_Gweiss_iw is None:
    dt.Gweiss_iw['up'] << inverse(iOmega_n - t**2*SemiCircular(2.0*t))
  else:
    print "starting from the provided initial_Gweiss_iw"
    dt.Gweiss_iw['up'] << initial_Gweiss_iw
  dt.get_Gweiss_tau()
  dt.get_H0()

  print "Done initializing, about to dump..."
  dt.dump('initial')

def ipt_dw_mf_actions(dt):
  def impurity(dt):
    if (dt.U!=0.0):
      dt.get_Sigma_imp_tau()
      dt.get_Sigma_imp_iw()

  def lattice(dt):      
    if dt.fixed_n:
      search_for_mu( dt.get_mu, dt.set_mu, dt.get_n, dt.n, dt.ph_symmetry ) 
    else:     
      print "fixed mu calculation, doing G"
      dt.n = dt.get_n() 
      print "n(G_loc) =",dt.n
    dt.get_F()
    #dt.get_H0() 
    
  def pre_impurity(dt):
    if dt.fixed_n and dt.ph_symmetry and dt.n==0.5:
      print "n(Gweiss)=", dt.get_n0()
    else:
      search_for_mu( dt.get_mu0tilde, dt.set_mu0tilde, dt.get_n0, dt.n, dt.ph_symmetry ) 
      #print "n(Gweiss)=", dt.get_n0()
      
     

  actions = [
    generic_action( 
      name = "pre_impurity",
      main = pre_impurity,
      mixers = [],#[lambda data, it: 0],
      cautionaries = [],#[lambda data, it: 0], allowed_errors = [],               
      printout = lambda data, it: 0,
      short_timings = True
    ),
    generic_action( 
      name = "impurity",
      main = impurity,
      mixers = [], #[lambda data, it: 0],
      cautionaries = [],#[lambda data, it: 0], allowed_errors = [],               
      printout = lambda data, it: 0,
      short_timings = True
    ),
    generic_action( 
      name = "lattice",
      main = lattice,
      mixers = [],#[lambda data, it: 0],
      cautionaries = [],#[lambda data, it: 0], allowed_errors = [],               
      #printout = lambda data, it: (data.dump(it) if (int(it[-3:])%5==0) else 0),
      printout = lambda data, it: data.dump(it),
      short_timings = True
    )
  ]

  monitors = [
    monitor(
      monitored_quantity = lambda: dt.G_loc_iw['up'].data[dt.niw,0,0].imag, 
      h5key = 'ImG_loc_iw0_vs_it', 
      archive_name = dt.archive_name
    ),
    monitor(
      monitored_quantity = lambda: dt.Sigma_imp_iw['up'].data[dt.niw,0,0].imag, 
      h5key = 'ImSigma_imp_iw0_vs_it', 
      archive_name = dt.archive_name
    ),
    monitor(
      monitored_quantity = lambda: dt.F, 
      h5key = 'F_vs_it', 
      archive_name = dt.archive_name
    ),
    monitor(
      monitored_quantity = lambda: dt.mu, 
      h5key = 'mu_vs_it', 
      archive_name = dt.archive_name
    ),
    monitor(
      monitored_quantity = lambda: dt.mu0tilde, 
      h5key = 'mu0tilde_vs_it', 
      archive_name = dt.archive_name
    )  
  ]

  convergers = [
    converger( monitored_quantity = lambda: dt.Sigma_imp_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_imp_iw'),
    converger( monitored_quantity = lambda: dt.G_loc_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_G_loc_iw'),
    converger( monitored_quantity = lambda: dt.F, accuracy=1e-4, func=converger.check_scalar, archive_name=dt.archive_name, h5key='diffs_F')
  ]

  return actions, monitors, convergers

def ipt_dw_mf_launcher( 
  n, mu, U, T, g, t=-0.25, initial_F=0, initial_Gweiss_iw=None,
  fixed_n = True,
  ph_symmetry = True,  
  nk=64,
  max_its = 10,
  min_its = 5, 
  iw_cutoff=30.0,
  filename=None
 ):
  print "------------------- Welcome to IPT + d-wave superconductivity at mean-field level! -------------------------"
  beta = 1.0/T
  niw = int(((iw_cutoff*beta)/math.pi-1.0)/2.0)
  ntau = 3*niw
  print "Automatic niw:",niw
  dt = ipt_dw_mf_data( niw, ntau, nk, beta )  
  ipt_dw_mf_set_calc(dt)
  ipt_dw_mf_set_params_and_initialize(dt, n, mu, g, T, U, t, fixed_n, ph_symmetry, initial_F, initial_Gweiss_iw, filename)
  actions, monitors, convergers = ipt_dw_mf_actions(dt)

  dmft = generic_loop(
    name = "IPT_DW_MF", 
    actions = actions,
    convergers = convergers,  
    monitors = monitors
  )

  dmft.run(
    dt, 
    max_its = max_its,
    min_its = min_its, 
    max_it_err_is_allowed = 7,
    print_final = True,
    print_current = 1,
    start_from_action_index = 1
  )

  return dt
