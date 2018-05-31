from smart_scripts import *
from functools import partial

def iaipt_data( niw, ntau, ks, field_ids, T ):
  dt = data() 
  dt.niw = niw
  dt.ntau = ntau
  dt.nk = len(ks)
  dt.ks = ks
  dt.nfields = len(field_ids)
  dt.field_ids = field_ids
  dt.beta = 1.0/T
  dt.T = T
  dt.iws = numpy.array([1j*(2.0*n+1)*pi*dt.T for n in range(-niw,niw)])
  print "Adding loc, imp and n.n. anomalous Green's functions..."  
  dt.G_loc_iw = GfImFreq(indices = range(dt.nfields), beta = dt.beta, n_points = niw, statistic = 'Fermion')
  dt.G_loc_tau = GfImTime(indices = range(dt.nfields), beta = dt.beta, n_points = ntau, statistic = 'Fermion')
  AddGfData(dt, ['Sigma_imp_iw','Gweiss_iw'], field_ids, 1, niw, dt.beta, domain = 'iw', suffix='', statistic='Fermion')
  AddGfData(dt, ['Sigma_imp_tau','Gweiss_tau'], field_ids, 1, ntau, dt.beta, domain = 'tau', suffix='', statistic='Fermion')

  print "Adding lattice Green's functions..."  
  dt.G_ab_k_iw =  numpy.zeros((2*niw, dt.nk,dt.nk, dt.nfields, dt.nfields), dtype=numpy.complex_)
  dt.H0k =  numpy.zeros( (dt.nk,dt.nk, dt.nfields,dt.nfields), dtype=numpy.complex_ )
  for Q in ['Us','ns','mu0tildes','Bs']:
    vars(dt)[Q] = dict.fromkeys(field_ids,0)
  print "Done preparing containers"  
  return dt

def iaipt_set_calc(
  dt, 
  get_H0k,
  starting_iw
):
  def fill_H0k(mu):
    dt.H0k[:,:,:,:] = get_H0k(mu)
  dt.get_H0k = lambda: fill_H0k(dt.mu)

  def get_G():
    for iwi, iw in enumerate(dt.iws):
      for kxi,kx in enumerate(dt.ks):
        for kyi,ky in enumerate(dt.ks):
          dt.G_ab_k_iw[iwi,kxi,kyi,:,:] = numpy.linalg.inv(\
            iw*numpy.eye(dt.nfields)\
            - dt.H0k[kxi,kyi,:,:]\
            - numpy.diag( [Sigma.data[iwi,0,0] for field_id, Sigma in dt.Sigma_imp_iw] )\
          )
        
  dt.get_G = lambda: get_G()
  
  def get_G_loc():
    dt.G_loc_iw.data[:,:,:] = numpy.sum(dt.G_ab_k_iw[:,:,:,:,:],axis=(1,2))/(dt.nk**2)
    fit_fermionic_gf_tail(dt.G_loc_iw,starting_iw=starting_iw)
    dt.G_loc_tau << InverseFourier(dt.G_loc_iw)
  dt.get_G_loc = lambda: get_G_loc()
 
  dt.get_mu = lambda: dt.mu

  def set_mu(mu):
    dt.mu = mu
  dt.set_mu = lambda mu: set_mu(mu)

  dt.get_n = lambda: [
    dt.get_H0k(), 
    dt.get_G(),
    dt.get_G_loc(),
    -numpy.sum(numpy.diag(dt.G_loc_tau.data[-1,:,:].real))
  ][-1]

  dt.get_ns = lambda: { fid: -dt.G_loc_tau.data[-1,fidi,fidi].real for fidi, fid in enumerate(dt.field_ids) }

  dt.get_Gweiss_iw = lambda field_id: orbital_space_dyson_get_G0(
    dt.Gweiss_iw[field_id],
    dt.G_loc_iw[dt.field_ids.index(field_id),dt.field_ids.index(field_id)],
    dt.Sigma_imp_iw[field_id],
    dt.mu0tildes[field_id]
  )
    
  def get_Gweiss_tau(field_id):
    impose_real_valued_in_imtime(dt.Gweiss_iw[field_id])  
    fit_fermionic_gf_tail(dt.Gweiss_iw[field_id])
    dt.Gweiss_tau[field_id] << InverseFourier(dt.Gweiss_iw[field_id]) 
  dt.get_Gweiss_tau = lambda field_id: get_Gweiss_tau(field_id)

  dt.get_mu0tilde = lambda field_id: dt.mu0tildes[field_id]
  def set_mu0tilde(mu0tilde, field_id):
    dt.mu0tildes[field_id] = mu0tilde
  dt.set_mu0tilde = lambda mu0tilde, field_id: set_mu0tilde(mu0tilde,field_id)
  dt.get_n0 = lambda field_id: [
    dt.get_Gweiss_iw(field_id),
    dt.get_Gweiss_tau(field_id),
    -dt.Gweiss_tau[field_id].data[-1,0,0].real
  ][-1]

  dt.get_Sigma_imp_tau = lambda field_id: get_Sigma_imp_tau_from_Gweiss_tau(dt.Sigma_imp_tau[field_id], dt.Gweiss_tau[field_id], dt.Us[field_id])

  dt.get_B = lambda n, U, mu0tilde: ((1.0-n)*U + mu0tilde)/(n*(1.0-n)*U**2)      
  def get_Sigma_imp_iw(field_id):
    dt.Bs[field_id] = dt.get_B(dt.ns[field_id], dt.Us[field_id], dt.mu0tildes[field_id])
    dt.Sigma_imp_iw[field_id] << Fourier(dt.Sigma_imp_tau[field_id])
    dt.Sigma_imp_iw[field_id] << dt.ns[field_id]*dt.Us[field_id] + dt.Sigma_imp_iw[field_id]*inverse(1.0-dt.Bs[field_id]*dt.Sigma_imp_iw[field_id])
    fit_fermionic_sigma_tail(dt.Sigma_imp_iw[field_id], starting_iw=starting_iw)
  dt.get_Sigma_imp_iw = lambda field_id: get_Sigma_imp_iw(field_id)

def iaipt_actions(dt, accr):
  def impurity(dt):
    for field_id in dt.field_ids: 
      if (dt.Us[field_id]!=0.0):
        dt.get_Sigma_imp_tau(field_id)
        dt.get_Sigma_imp_iw(field_id)

  def lattice(dt):      
    search_for_mu( 
      dt.get_mu,
      dt.set_mu,
      dt.get_n,
      dt.n,
      False
    ) 
    print "n(G_loc) =",dt.get_n()
    dt.ns = dt.get_ns()
    
  def pre_impurity(dt):
    for field_id in dt.field_ids:
      if (dt.Us[field_id]!=0.0):
        search_for_mu( 
          partial(dt.get_mu0tilde, field_id=field_id),
          partial(dt.set_mu0tilde, field_id=field_id),
          partial(dt.get_n0, field_id=field_id),
          dt.ns[field_id],
          False
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
      #printout = lambda data, it: (data.dump(it) if (int(it[-3:])%5==0) else 0),
      printout = lambda data, it: data.dump(it),
      short_timings = True
    )
  ]

  monitors = [
    monitor(
      monitored_quantity = lambda: dt.G_loc_iw.data[dt.niw,0,0].imag, 
      h5key = 'ImG_loc_iw0_vs_it', 
      archive_name = dt.archive_name
    ),
    monitor(
      monitored_quantity = lambda: dt.mu, 
      h5key = 'mu_vs_it', 
      archive_name = dt.archive_name
    )
  ]
  monitors.extend( [
    monitor(
      monitored_quantity = lambda field_id=field_id: dt.Sigma_imp_iw[field_id].data[dt.niw,0,0].imag, 
      h5key = 'ImSigma_imp-%s_iw0_vs_it'%field_id, 
      archive_name = dt.archive_name
    ) for field_id in dt.field_ids
  ] )
  monitors.extend( [
    monitor(
      monitored_quantity = lambda field_id=field_id: dt.mu0tildes[field_id], 
      h5key = 'mu0tilde-%s_vs_it'%field_id, 
      archive_name = dt.archive_name
    ) for field_id in dt.field_ids
  ] )

  convergers = [
    #converger( monitored_quantity = lambda: dt.Sigma_imp_iw, accuracy=accr, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_imp_iw'),
    converger( monitored_quantity = lambda: dt.G_loc_iw, accuracy=accr, func=None, archive_name=dt.archive_name, h5key='diffs_G_loc_iw')
  ]

  return actions, monitors, convergers

def iaipt_set_params_and_initialize(
  dt,
  n, 
  Us, 
  initial_mu,
  initial_guess,
  filename
):
  dt.archive_name = filename
  dt.dump = lambda dct: DumpData(dt, filename, Qs=[], exceptions=['G_ab_k_iw'], dictionary=dct)
  dt.dump_final = lambda dct: DumpData(dt, filename, Qs=[], exceptions=[], dictionary=dct)

  dt.n = n #if mu is fized, n is used as initial guess
  dt.mu = 0.0#initial_mu #if fixed_n, mu is used as initial 
  dt.get_H0k()
  for field_id in dt.field_ids:
    dt.mu0tildes[field_id] = 0 
    try: dt.Us[field_id] = Us[field_id]
    except: dt.Us[field_id] = 0.0 
    dt.Sigma_imp_iw[field_id] << float((initial_guess=='insulator')and(dt.Us[field_id]!=0.0))*inverse(iOmega_n)
  dt.get_n()
  dt.ns = dt.get_ns()
  dt.mu = dt.Us[dt.field_ids[0]]/2.0 #!!!!!!!!!!1
  #dt.mu0tildes = dict.fromkeys(dt.field_ids, -initial_mu) 
  print "Done initializing, about to dump..."
  dt.dump('initial')


def iaipt_launcher( 
  field_ids,
  get_H0k,
  Us,
  n, 
  T, 
  starting_iw,
  ks,
  initial_guess,
  initial_mu,
  max_its,
  min_its, 
  accr,
  filename
):
  print "------------------- Welcome to IAIPT! -------------------------"
  print "field_ids:",field_ids
  print "Us:",Us
  print "n:",n
  print "T:",T
  print "accr:",accr

  beta = 1.0/T
  iw_cutoff= 2.0*starting_iw
  niw = int(((iw_cutoff*beta)/math.pi-1.0)/2.0)
  ntau = 3*niw
  print "Automatic niw:",niw
  dt = iaipt_data( niw, ntau, ks, field_ids, T )  
  iaipt_set_calc(dt, get_H0k, starting_iw)
  iaipt_set_params_and_initialize( dt, n, Us, initial_mu, initial_guess, filename)
  actions, monitors, convergers = iaipt_actions(dt, accr)

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
    start_from_action_index = 0
  )

  return dt, convergers
