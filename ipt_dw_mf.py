def ipt_dw_mf_data( niw, ntau, nk, beta ):
  dt = data() 
  dt.niw = niw
  dt.ntau = ntau
  dt.nk = nk
  dt.beta = beta
  dt.T = 1.0/beta
  dt.iws = [1j*(2.0*n+1)*pi*dt.T for n in range(-niw,niw)]
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
  gk = lambda kx,ky, g: g*numpy.cos(kx)-numpy.cos(ky)  
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

  dt.get_G = lambda: get_G()

  def get_G_loc():
    for iwi, iw in enumerate(dt.iws):
      dt.G_loc_iw['up'].data[iwi,0,0] = numpy.sum(dt.G_ab_k_iw[iwi,0,0,:,:])/dt.nk**2
    fit_fermionic_gf_tail(dt.G_loc_iw['up'])
    dt.G_loc_tau << InverseFourier(dt.G_loc_iw)
  dt.get_G_loc = lambda: get_G_loc()
 
  dt.get_mu = lambda: dt.mu

  def set_mu(mu):
    dt.mu = mu
  dt.set_mu = lambda mu: set_mu(mu)

  dt.get_n = lambda: [ dt.get_G_loc(), -dt.G_loc_tau['up'].data[-1,0,0].real][1]

  def get_F():
    F_k_iw[:,:,:] = dt.G_ab_k_iw[:,:,:,0,1]
    F_r_iw[:,:,:] =  [ numpy.fft.ifft2(F_k_iw[l,:,:]) for l in range(n)]
    F_r10_iw['up'].data[:,0,0] = F_r_iw[:,1,0]
    F_r10_tau << InverseFourier(F_r10_iw)
    dt.F = F_r10_tau['up'].data[0,0,0].real 
  
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
    dt.Gweiss_tau << InverseFourier(dt.Gweiss_iw) 
  dt.get_Gweiss_tau = lambda: dt.get_Gweiss_tau()

  dt.get_mu0tilde = lambda: dt.mu0tilde
  def set_mu0tilde(mu0tilde):
    dt.mu0tilde = mu0tilde
  dt.set_mu0tilde = lambda mu0tilde: set_mu0tilde(mu0tilde)
  dt.get_n0 = lambda: [
    dt.get_Gweiss_iw(dt.mu0tilde),
    dt.get_Gweiss_tau(),
    -dt.Gweiss_tau['up'].data[-1,0,0].real
  ][2]

  dt.get_Sigma_imp_tau = lambda: get_Sigma_imp_tau_from_Gweiss_tau(dt.Sigma_imp_tau, dt.Gweiss_tau, dt.U)

  dt.get_B = lambda n, U, mu0tilde: ((1-n)*U + mu0tilde)/(n*(1-n)*U**2)      
  def get_Sigma_imp_iw():      
      dt.B = dt.get_B(dt.n, dt.U, dt.mu0tilde)
      dt.Sigma_imp_iw << Fourier(dt.Sigma_imp_tau)
      dt.Sigma_imp_iw << dt.n*dt.U + dt.Sigma_imp_iw*inverse(1.0-dt.B*dt.Sigma_imp_iw)
      fit_fermionic_sigma_tail(dt.Sigma_imp_iw['up'])
  dt.get_Sigma_imp_iw = lambda: get_Sigma_imp_iw()

def ipt_dw_mf_set_params_and_initialize(dt, n, mu, g, T, U, fixed_n = True, ph_symmetry = True, initial_F=0.0, filename=None):
  if filename is None: 
    filename = "ipt_dw_mf"
    if fixed_n:
      filename += ".n%.4f"%n
    else:
      filename += ".mu%.4f"%mu
    filename += "U%.4f.T%.4f"%(dt.U,dt.T)
    if initial_F!=0.0: filename+=".from_sc"
    else: filename+=".from_normal"

  dt.archive_name = filename
  dt.dump = lambda dct: DumpData(dt, filename, Qs=[], exceptions=[], dictionary=dct)
  dt.dump_final = lambda dct: DumpData(dt, filename, Qs=[], exceptions=[], dictionary=dct)

  dt.ks = numpy.linspace(0,2.0*numpy.pi,dt.nk,endpoint=False)
  dt.n = n
  dt.mu = mu #if fixed_n, mu is used as initial guess
  dt.mu0tilde = 0
  dt.U = U
  dt.g =g
  dt.t = t    
  dt.F = initial_F
  dt.fixed_n = fixed_n
  dt.ph_symmetry = ph_symmetry

  dt.get_H0()

  print "Done initializing, about to dump..."
  dt.dump('initial')

def ipt_dw_mf_actions(dt):
  def impurity(dt):
    if (dt.U!=0.0):
      dt.get_Sigma_imp_tau()
      dt.get_Sigma_imp_iw()

  def lattice(dt):   
    dt.get_F()
    dt.get_H0() 
    
    if dt.fixed_n:
      search_for_mu(
        (lambda: dt.get_mu()),
        (lambda: dt.set_mu()),
        (lambda: get_n()), 
        dt.n, 
        dt.ph_symmetry
      ) 
    else: 
      dt.get_G()      
      dt.n = dt.get_n() 
    
  def pre_impurity(dt):
    if fixed_n and ph_symmetry and dt.n==0.5:
      dt.get_Gweiss_iw(0.0)
      dt.get_Gweiss_tau()
    else:
      search_for_mu(
        (lambda: dt.get_mu0tilde()),
        (lambda: dt.set_mu0tilde()),
        (lambda: get_n0()), 
        dt.n, 
        dt.ph_symmetry
      ) 

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
      printout = lambda data, it: (data.dump(it) if (int(it[-3:])%5==0) else 0),
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
    ) 
  ]

  convergers = [
    converger( monitored_quantity = lambda: dt.Sigma_imp_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_imp_iw')
    converger( monitored_quantity = lambda: dt.G_loc_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_G_loc_iw')
    converger( monitored_quantity = lambda: dt.F, accuracy=1e-4, func=converger.check_scalar, archive_name=dt.archive_name, h5key='diffs_F')
  ]

  return actions, monitors, convergers

def ipt_dw_mf_launcher( 
  n, mu, U, T, g, t=-0.25, initial_F=0,
  fixed_n = True,
  ph_symmetry = True  
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
  ipt_dw_mf_set_params_and_initialize(dt, n, mu, g, T, U, fixed_n, ph_symmetry, initial_F)
  actions, monitors, convergers = ipt_dw_mf_actions(dt)

  rdmft = generic_loop(
    name = "RDMFT", 
    actions = actions,
    convergers = convergers,  
    monitors = monitors
  )

  rdmft.run(
    dt, 
    max_its = max_its,
    min_its = min_its, 
    max_it_err_is_allowed = 7,
    print_final = True,
    print_current = -1,
    start_from_action_index = 0
  )

  return dt
