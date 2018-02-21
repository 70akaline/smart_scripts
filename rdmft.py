from smart_scripts import *

def rdmft_data( niw, ntau, nsites, beta, blocks = ['up'] ):
  dt = data() 
  dt.niw = niw
  dt.ntau = ntau
  dt.nsites = nsites
  dt.beta = beta
  dt.blocks = blocks 
  print "Adding lattice Green's functions..."  
  #AddGfData(dt, ['G_ij_iw', 'Sigma_ij_iw', 'G0_ij_iw'], blocks, nsites, niw, beta, domain = 'iw', suffix='', statistic='Fermion')
  #AddGfData(dt, ['G_ij_iw', 'G0_ij_iw'], blocks, nsites, niw, beta, domain = 'iw', suffix='', statistic='Fermion') #optimizing, we don't really need Sigma_ij_iw
  AddGfData(dt, ['G_ij_iw'], blocks, nsites, niw, beta, domain = 'iw', suffix='', statistic='Fermion') #optimizing further, we don't even need G0_ij_iw
  dt.combo_blocks = [ "%s|%.3d"%(block,i) for i in range(nsites) for block in blocks]
  print "Adding imp Green's functions..."  
  AddGfData(dt, ['Sigma_imp_iw','Gweiss_iw'], dt.combo_blocks, 1, niw, beta, domain = 'iw', suffix='', statistic='Fermion')
  AddGfData(dt, ['Sigma_imp_tau','Gweiss_tau'], dt.combo_blocks, 1, ntau, beta, domain = 'tau', suffix='', statistic='Fermion')
  print "Adding imp numpy arrays..."  
  AddBlockNumpyData(dt, ['H0'], blocks, (nsites,nsites))
  AddBlockScalarData(dt, dt.combo_blocks, ['mus','Us'], vals=None)
  print "Done preparing containers"  
  return dt

def rdmft_set_params_and_initialize(dt, Nx, Ny, U, T, C, t=-0.25, initial_guess='metal', filename=None):
  if filename is None: 
    filename = "rdmft.%dx%d.U%.4f.T%.4f.C%.4f.from_%s"\
                %(Nx,Ny,U,T,C,initial_guess)
  dt.archive_name = filename
  dt.dump = lambda dct: DumpData(dt, filename, Qs=[], exceptions=['G_ij_iw'], dictionary=dct)
  dt.dump_final = lambda dct: DumpData(dt, filename, Qs=[], exceptions=[], dictionary=dct)  

  def get_block_from_combo_block(cb):
    return cb.split('|')[0]
  dt.get_block_from_combo_block = get_block_from_combo_block

  def get_site_index_from_combo_block(cb):   
    cbi = dt.combo_blocks.index(cb)
    return cbi % dt.nsites
  dt.get_site_index_from_combo_block = get_site_index_from_combo_block

  nsites = Nx*Ny
  dt.Us_array = {}
  for b in dt.blocks:
    print "Making H0..."
    dt.H0[b][:,:] = initCubicTBH(Nx, Ny, 1, 0, t, cyclic=True)
    iws = numpy.array([iw.value for iw in dt.G_ij_iw[b].mesh])     
    print "Making G0, storing in G..."
    for i in range(dt.nsites):
      dt.G_ij_iw[b].data[:,i,i] = iws
    dt.G_ij_iw[b].data[:,:,:] -= dt.H0[b]
    for iwi, iw in enumerate(iws):  
      dt.G_ij_iw[b].data[iwi,:,:] = numpy.linalg.inv(dt.G_ij_iw[b].data[iwi,:,:])
    print "Fitting tail to G0..."
    fit_fermionic_gf_tail(dt.G_ij_iw[b], starting_iw=14.0, no_loc=False, overwrite_tail=False, max_order=5)
    print "Choosing Us..."
    dt.Us_array[b] = numpy.random.choice([U,0], size=nsites, p=[C, 1-C]) 
  #dt.Sigma_ij_iw << 0.0
  dt.Sigma_imp_iw << 0.0
  #dt.G_ij_iw << dt.G0_ij_iw

  for cb in dt.combo_blocks:
    block = get_block_from_combo_block(cb)
    site_index = get_site_index_from_combo_block(cb)
    dt.mus[cb] = 0.0
    dt.Us[cb] = dt.Us_array[block][site_index]
    if initial_guess=='metal': dt.Gweiss_iw[cb] << dt.G_ij_iw[block][site_index,site_index]
    elif initial_guess=='atomic': dt.Gweiss_iw[cb] << inverse(iOmega_n)
    else: assert False, 'initial guess '+str(initial_guess)+' not implemented!'
  print "Done initializing, about to dump..."
  dt.dump('initial')

def rdmft_set_calc(dt):
  dt.get_Sigma_imp_tau = lambda block: get_Sigma_imp_tau_from_Gweiss_tau(
    dt.Sigma_imp_tau[block], 
    dt.Gweiss_tau[block],
    dt.Us[block]
  )

  def get_Sigma_imp_iw(cb):
    dt.Sigma_imp_iw[cb] << Fourier(dt.Sigma_imp_tau[cb]) 
  dt.get_Sigma_imp_iw = get_Sigma_imp_iw
  
  #def get_Sigma_ij_iw():
  #  for cb,sig in dt.Sigma_imp_iw:
  #    #print "cb:",cb
  #    b = dt.get_block_from_combo_block(cb)
  #    i = dt.get_site_index_from_combo_block(cb)
  #    #print "b,i:",b,i
  #    dt.Sigma_ij_iw[b][i,i] << sig
  #dt.get_Sigma_ij_iw = get_Sigma_ij_iw

  def memory_optimized_orbital_space_dyson():
    for b in dt.blocks:
      iws = numpy.array([iw.value for iw in dt.G_ij_iw[b].mesh]) 
      for iwi, iw in enumerate(iws):
        #dt.G_ij_iw[b].data[iwi,:,:] = numpy.linalg.inv(dt.G0_ij_iw[b].data[iwi,:,:])
        dt.G_ij_iw[b].data[iwi,:,:] = -dt.H0[b][:,:]
      for i in range(dt.nsites):
        dt.G_ij_iw[b].data[:,i,i] += iws - dt.Sigma_imp_iw["%s|%.3d"%(b,i)].data[:,0,0]
      for iwi, iw in enumerate(iws):
        dt.G_ij_iw[b].data[iwi,:,:] = numpy.linalg.inv(dt.G_ij_iw[b].data[iwi,:,:]) 
      fit_fermionic_gf_tail(dt.G_ij_iw[b], starting_iw=14.0, no_loc=False, overwrite_tail=False, max_order=5)
  #dt.get_G_ij_iw = lambda: orbital_space_dyson_get_G(dt.G_ij_iw, dt.G0_ij_iw, dt.Sigma_ij_iw)
  dt.get_G_ij_iw = memory_optimized_orbital_space_dyson  

  dt.get_Gweiss_iw = lambda cb: orbital_space_dyson_get_G0(
    dt.Gweiss_iw[cb],
    dt.G_ij_iw[dt.get_block_from_combo_block(cb)][dt.get_site_index_from_combo_block(cb),dt.get_site_index_from_combo_block(cb)],
    dt.Sigma_imp_iw[cb]
  )

  def get_Gweiss_tau(cb):
    fit_fermionic_gf_tail(dt.Gweiss_iw[cb], starting_iw=14.0, no_loc=False, overwrite_tail=True, max_order=5)
    dt.Gweiss_tau[cb] << InverseFourier(dt.Gweiss_iw[cb]) 
  dt.get_Gweiss_tau = get_Gweiss_tau

def rdmft_actions(dt):
  def impurity(dt):
    for cb in dt.combo_blocks:
      if dt.Us[cb]==0.0: continue
      dt.get_Sigma_imp_tau(cb)
      dt.get_Sigma_imp_iw(cb)

  def lattice(dt):
    #dt.get_Sigma_ij_iw()
    dt.get_G_ij_iw()

  def pre_impurity(dt):
    for cb in dt.combo_blocks:
       dt.get_Gweiss_iw(cb) 
       dt.get_Gweiss_tau(cb)

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
      printout = lambda data, it: data.dump(it),
      short_timings = True
    )
  ]

  monitors = [
#    monitor(
#      monitored_quantity = lambda i=i: dt.G_ij_iw[dt.blocks[0]].data[dt.niw,i,i].imag, 
#      h5key = 'ImG_%s%s_iw_0_vs_it'%(i,i), 
#      archive_name = dt.archive_name
#    ) for i in range(dt.nsites)[]
  ]

  convergers = [
    #converger( monitored_quantity = lambda: dt.G_ij_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_G_ij_iw'),
    #converger( monitored_quantity = lambda: dt.Sigma_ij_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_ij_iw')
    converger( monitored_quantity = lambda: dt.Sigma_imp_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_imp_iw')
  ]

  return actions, monitors, convergers

def rdmft_launcher(Nx, Ny, U, T, C,
                   t=-0.25, 
                   initial_guess='metal',
                   max_its = 10,
                   min_its = 5, 
                   iw_cutoff=30.0,
                   filename=None ):
  print "------------------- Welcome to RDMFT! -------------------------"
  beta = 1/T
  niw = int(((iw_cutoff*beta)/math.pi-1.0)/2.0)
  ntau = 3*niw
  print "Automatic niw:",niw
  
  nsites = Nx*Ny 
  dt = rdmft_data( niw, ntau, nsites, beta )
  rdmft_set_params_and_initialize(dt, Nx, Ny, U, T, C, t, initial_guess, filename)
  rdmft_set_calc(dt)
  actions, monitors, convergers = rdmft_actions(dt)

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
    print_current = 5,
    start_from_action_index = 1
  )

  return dt
