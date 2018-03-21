from smart_scripts import *

def dw_mf_data( niw, ntau, nk, beta ):
  dt = data() 
  dt.niw = niw
  dt.ntau = ntau
  dt.nk = nk
  dt.beta = beta
  dt.T = 1.0/beta
  dt.iws = [1j*(2.0*n+1)*pi*dt.T for n in range(-niw,niw)]
  print "Adding lattice Green's functions..."  
  AddNumpyData(dt, ['G_ab_k_iw'], (2*niw, nk,nk, 2,2) )
  AddNumpyData(dt, ['H0'], (nk,nk,2,2))
  AddNumpyData(dt, ['ks'], (nk))
  AddScalarData(dt, ['mu','F','g'], vals=None)
  print "Done preparing containers"  
  return dt

def dw_mf_set_calc(dt):
  def H0(kx,ky,t, mu,g,F):
    epsk = 2*t*(numpy.cos(kx)+numpy.cos(ky))
    dwk = numpy.cos(kx)-numpy.cos(ky)  
    return numpy.array([
      [ epsk-mu, g*F*dwk ],
      [ g*F*dwk, mu-epsk ],
    ])
 
  def fill_H0(): 
    for kxi,kx in enumerate(dt.ks):
      for kyi,ky in enumerate(dt.ks):
        dt.H0[kxi,kyi,:,:] = H0(kx,ky,dt.t,dt.mu,dt.g,dt.F)
    
  dt.get_H0 = lambda: fill_H0()

  def get_G():
    for iwi, iw in enumerate(dt.iws):
      for kxi,kx in enumerate(dt.ks):
        for kyi,ky in enumerate(dt.ks):
          dt.G_ab_k_iw[iwi,kxi,kyi,:,:] = numpy.linalg.inv(iw*numpy.array([[1,0],[0,1]])-dt.H0[kxi,kyi])

  dt.get_G = lambda: get_G()

  def get_F():
    F_k_iw = dt.G_ab_k_iw[:,:,:,0,1]
    n,nk,nk = numpy.shape(F_k_iw)
    F_r_iw = numpy.zeros((n,nk,nk), dtype=numpy.complex_)          
    F_r_iw[:,:,:] =  [ numpy.fft.ifft2(F_k_iw[l,:,:]) for l in range(n)]
    F_r10_iw = GfImFreq(indices = [0], beta = dt.beta, n_points = dt.niw, statistic = 'Fermion')
    F_r10_iw.data[:,0,0] = F_r_iw[:,1,0]
    F_r10_tau = GfImTime(indices = [0], beta = dt.beta, n_points = dt.ntau, statistic = 'Fermion')
    F_r10_tau << InverseFourier(F_r10_iw)
    dt.F = F_r10_tau.data[0,0,0].real 
  
  dt.get_F = lambda: get_F()

def dw_mf_set_params_and_initialize(dt, mu, g, initial_F=0.0, t=-0.25, filename=None):
  if filename is None: 
    filename = "dw_mf.mu%.4f.T%.4f"\
                %(mu,dt.T)
  dt.archive_name = filename
  dt.dump = lambda dct: DumpData(dt, filename, Qs=[], exceptions=[], dictionary=dct)
  dt.dump_final = lambda dct: DumpData(dt, filename, Qs=[], exceptions=[], dictionary=dct)

  dt.ks = numpy.linspace(0,2.0*numpy.pi,dt.nk,endpoint=False)
  dt.F = initial_F
  dt.g = g
  dt.mu = mu
  dt.t = t
  dt.get_H0()

  print "Done initializing, about to dump..."
  dt.dump('initial')

def dw_mf_actions(dt):
  actions = [
    generic_action( 
      name = "get G",
      main = lambda dt: dt.get_G(),
      mixers = [],#[lambda data, it: 0],
      cautionaries = [],#[lambda data, it: 0], allowed_errors = [],               
      printout = lambda data, it: 0,
      short_timings = True
    ),
    generic_action( 
      name = "get F",
      main = lambda dt: dt.get_F(),
      mixers = [], #[lambda data, it: 0],
      cautionaries = [],#[lambda data, it: 0], allowed_errors = [],               
      printout = lambda data, it: 0,
      short_timings = True
    ),
    generic_action( 
      name = "get H0",
      main = lambda dt: dt.get_H0(),
      mixers = [],#[lambda data, it: 0],
      cautionaries = [],#[lambda data, it: 0], allowed_errors = [],               
      printout = lambda data, it: (data.dump(it) if (int(it[-3:])%5==0) else 0),
      short_timings = True
    )
  ]

  monitors = [
    monitor(
      monitored_quantity = lambda: dt.F, 
      h5key = 'F_vs_it', 
      archive_name = dt.archive_name
    )
  ]

  convergers = [
    converger( monitored_quantity = lambda: dt.G_ab_k_iw, accuracy=1e-4, func=converger.check_numpy_array, archive_name=dt.archive_name, h5key='diffs_G_ab_k_iw'),
    converger( monitored_quantity = lambda: dt.F, accuracy=1e-4, func=converger.check_scalar, archive_name=dt.archive_name, h5key='diffs_F')
  ]

  return actions, monitors, convergers

def dw_mf_launcher(nk, mu, T, g,
                   t=-0.25, 
                   initial_F=0,
                   max_its = 10,
                   min_its = 5, 
                   iw_cutoff=30.0,
                   filename=None ):
  print "------------------- Welcome to d-weave mean-field! -------------------------"
  beta = 1.0/T
  niw = int(((iw_cutoff*beta)/math.pi-1.0)/2.0)
  ntau = 3*niw
  print "Automatic niw:",niw
  dt = dw_mf_data( niw, ntau, nk, beta )  
  dw_mf_set_calc(dt)
  dw_mf_set_params_and_initialize(dt, mu, g, initial_F, t, filename)
  actions, monitors, convergers = dw_mf_actions(dt)

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
    print_current = 100000,
    start_from_action_index = 0
  )

  return dt
