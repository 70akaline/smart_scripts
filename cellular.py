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

def get_nns(Nx,Ny,l): #find linear indices of the 4 nearest neighbors of site with linear index l ona cyclic square cluster Nx x Ny
  i = l % Nx
  j = int(l / Nx)
  lrn, lln, lbn, ltn = l+1,l-1,l+Nx,l-Nx
  if i+1==Nx: lrn = None
  if i==0: lln = None
  if j+1==Ny: lbn = None
  if j==0: ltn = None
  return [lrn,lln,lbn,ltn]

def initialize_F(Nx,Ny, value=1.0):
  nsites = Nx*Ny 
  F = numpy.zeros((nsites,nsites))
  for l in range(nsites):
    lnns = get_nns(Nx,Ny,l)
    for lnni,lnn in enumerate(lnns):
      if lnn is None: continue
      F[l,lnn] = (1 if (lnni<2) else -1)*value
  return F

def fill_H0_k(H0_k,Nx,Ny,t,ks, mus, F=None):
  nsites = Nx*Ny 
  for kxi,kx in enumerate(ks):
    for kyi,ky in enumerate(ks):
      epsk = matrix_dispersion(Nx,Ny,t,kx,ky)
      H0_k[kxi,kyi,:nsites,:nsites] = epsk - numpy.diag(mus) 
      H0_k[kxi,kyi,nsites:,nsites:] = numpy.diag(mus) - epsk
      if not(F is None):
        H0_k[kxi,kyi,:nsites,nsites:] = F
        H0_k[kxi,kyi,nsites:,:nsites] = F


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

def get_ns(G_IaJb_loc_tau):
  ntau, nIa, nJb = numpy.shape(G_IaJb_loc_tau.data)
  tot = 0
  nsites = nIa/2
  ns = {'up':[],'dn':[]}
  for Ia in range(nsites):
      ns['up'].append(-G_IaJb_loc_tau.data[0,Ia,Ia])
      ns['dn'].append(-G_IaJb_loc_tau.data[-1,Ia+nsites,Ia+nsites])
  return ns

def get_Gweiss(Gweiss_IaJb_iw, G_IaJb_loc_iw, Sigma_IaJb_imp_iw):
  Gweiss_IaJb_iw << inverse(inverse(G_IaJb_loc_iw)+Sigma_IaJb_imp_iw)
  impose_real_valued_in_imtime(Gweiss_IaJb_iw)
  fit_fermionic_gf_tail(Gweiss_IaJb_iw, starting_iw=14.0, no_loc=False, overwrite_tail=True, max_order=5)

def get_Sigma(solver, Sigma_imp_iw, Gweiss_iw, Us, max_time=5*60, delta=0.1, nambu=False, solver_data_package=None, su2=True, nambu=False):
  if nambu:
    solver.G0_iw['nambu'] << Gweiss_iw
  else: assert False, "not implemented..."

  solvers.ctint.run(
    solver=solver, 
    Us=Us,      
    nambu=nambu,
    alpha=0.5,
    delta=delta,
    n_cycles=20000,
    max_time = max_time,
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

def cellular_set_calc( dt, max_time=5*60, delta=0.1, solver_data_package=None, nambu=True, su2=True ):
  dt.get_H0_k = lambda: fill_H0_k(dt.H0_k,dt.Nx,dt.Ny,dt.t,dt.ks, dt.mus)

  dt.get_Gweiss = lambda: get_Gweiss(dt.Gweiss_IaJb_iw, dt.G_IaJb_loc_iw, dt.Sigma_IaJb_imp_iw)

  dt.get_Sigma = lambda: get_Sigma(dt.solver, dt.Sigma_IaJb_imp_iw, dt.Gweiss_IaJb_iw, dt.Us, max_time, delta, dt.nambu, solver_data_package, su2, nambu)

  dt.get_G = lambda: get_G(dt.G_IaJb_k_iw, dt.H0_k, dt.Sigma_IaJb_imp_iw)

  dt.get_G_loc = lambda: get_G_loc(dt.G_IaJb_loc_iw, dt.G_IaJb_k_iw)

  dt.get_mu = lambda: dt.mus[0]-dt.initial_mus[0]

  def set_mu(mu):
    dt.mus = dt.mus_initial+mu
    dt.get_H0_k()

  dt.set_mu = lambda mu: set_mu(mu)

  dt.get_n = lambda: [dt.get_G(), dt.get_G_loc(), get_n(dt.G_IaJb_loc_tau)][-1]

  dt.get_ns = lambda: get_ns(dt.G_IaJb_loc_tau)

def cellular_set_params_and_initialize(
  dt, Us, mus,
  n=None, fixed_n=False, ph_symmetry=True,
  t=-0.25,
  initial_guess='metal', initial_F=0.0, initial_afm=0.0,
  filename=None,
  solver_data_package=None
):
  assert len(Us)==dt.nsites, "wrong number of Us!"
  dt.Us = Us
  U = numpy.max(Us)
  dt.C = (dt.nsites-numpy.count_nonzero(Us))/dt.nsites
  dt.t = t
  dt.mus = mus
  dt.initial_mus = numpy.array(mus)
  dt.fixed_n = fixed_n
  dt.n=n 
  if filename is None: 
    filename = "cellular.%dx%d.U%.4f.T%.4f.C%.4f.from_%s_%s_%s"\
                %(dt.Nx,dt.Ny,U,dt.T,dt.C,initial_guess,('sc' if (initial_F!=0.0) else 'normal'),('afm' if (initial_afm!=0.0) else 'pm'))
  dt.archive_name = filename
  dt.dump = lambda dct: DumpData(dt, filename, Qs=[], exceptions=['G_IaJb_k_iw'], dictionary=dct)
  dt.dump_final = lambda dct: DumpData(dt, filename, Qs=[], exceptions=[], dictionary=dct)

  dt.solver = solvers.ctint.initialize_solver(
    nambu=True,
    solver_data_package = solver_data_package,  
    nsites = dt.nsites,
    niw = dt.niw,
    ntau = max(dt.ntau, 2000)
  )

  print "Making H0.."
  if initial_F!=0.0:
    fill_H0_k(dt.H0_k,dt.Nx,dt.Ny,dt.t,dt.ks, dt.mus, F=initialize_F(dt.Nx,dt.Ny, value=initial_F))
  else: dt.get_H0_k()
  print "Filling Sigma_imp_iw.."
  dt.Sigma_imp_iw << 0.0
  for I,U in Us:
    dt.Sigma_imp_iw[I,I] << U*0.5+initial_afm-int(initial_guess=='atomic')*inverse(iOmega_n)
    dt.Sigma_imp_iw[I+dt.nsites,I+dt.nsites] << U*0.5-initial_afm-int(initial_guess=='atomic')*inverse(iOmega_n)
  
  print "Getting G.."
  dt.get_G()
  dt.get_G_loc()
  print "Getting Gweiss.."
  dt.get_Gweiss() 
  if initial_F!=0.0: dt.get_H0_k() #we don't need anymore the anomalous part in H0. It's only for the first iteration.
    
  print "Done initializing, about to dump..."
  dt.dump('initial')

def cellular_actions(dt):
  def impurity(dt):
    dt.get_Sigma())

  def lattice(dt):
    if dt.fixed_n:
      search_for_mu( dt.get_mu, dt.set_mu, dt.get_n, dt.n, dt.ph_symmetry ) 
    else:     
      print "fixed mu calculation, doing G"
      dt.n = dt.get_n() 
      print "n(G_loc) =",dt.n

  def pre_impurity(dt):
    dt.get_Gweiss() 

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
#    monitor(
#      monitored_quantity = lambda i=i: dt.G_ij_iw[dt.blocks[0]].data[dt.niw,i,i].imag, 
#      h5key = 'ImG_%s%s_iw_0_vs_it'%(i,i), 
#      archive_name = dt.archive_name
#    ) for i in range(dt.nsites)[]
    monitor(
      monitored_quantity = lambda: numpy.amax(numpy.abs(dt.G_IaJb_loc_tau.data[0,dt.nsites:,:dt.nsites])), 
      h5key = 'Fmax_vs_it', 
      archive_name = dt.archive_name
    ),
    monitor(
      monitored_quantity = lambda: numpy.sum(numpy.abs(dt.G_IaJb_loc_tau.data[0,dt.nsites:,:dt.nsites])), 
      h5key = 'Ftot_vs_it', 
      archive_name = dt.archive_name
    )
  ]

  convergers = [
    #converger( monitored_quantity = lambda: dt.G_ij_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_G_ij_iw'),
    #converger( monitored_quantity = lambda: dt.Sigma_ij_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_ij_iw')
    converger( monitored_quantity = lambda: dt.Sigma_IaJb_imp_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_imp_iw'),
    converger( monitored_quantity = lambda: dt.G_IaJb_loc_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_G_IaJb_loc_iw'),
  ]

  return actions, monitors, convergers

def cellular_launcher(Nx, Ny,
                   Us, T,
                   mus=None,
                   n=0.5, fixed_n=True,
                   t=-0.25, 
                   initial_guess='metal',
                   initial_F=1.0,
                   initial_afm=0.0,
                   nk=24,
                   n_cycles=100000,            
                   max_time_rules= [ [1, 5*60], [2, 20*60], [4, 80*60], [8, 200*60], [16,400*60] ], time_rules_automatic=False, exponent = 0.7, overall_prefactor=1.0, no_timing = False,
                   max_its = 10,
                   min_its = 5, 
                   iw_cutoff=30.0,
                   filename=None ):
  solver_data_package={}
  if mpi.is_master_node():
    print "------------------- Welcome to RDMFT! -------------------------"
    beta = 1.0/T
    niw = int(((iw_cutoff*beta)/math.pi-1.0)/2.0)
    ntau = 3*niw
    print "Automatic niw:",niw
    
    nsites = Nx*Ny
    Umax = numpy.max(Us) 
    in no_timing:
      max_time=-1
      print "no timing!!!"
    else:
      if time_rules_automatic:
        pref = ((beta/8.0)*Umax*nsites)**exponent #**1.2
        print "pref: ",pref 
        max_time = int(overall_prefactor*pref*5*60)
        print "max times automatic: ",max_time
      else:
        for r in max_time_rules: if r[0]<=nsites: max_time = r[1]
        print "max_time from rules: ",max_time


    if mus is None:
      mus = numpy.array(Us)/2.0 

    dt = cellular_data( nk, niw, ntau, Nx,Ny, beta )
    cellular_set_calc( dt, max_time, delta=(0.1 if n==0.5 else 0.5), solver_data_package=solver_data_package, nambu=True, su2=(initial_afm==0.0) )
    cellular_set_params_and_initialize(
      dt, Us=Us, mus=mus,
      n=n, fixed_n=fixed_n, ph_symmetry=True,
      t=t,
      initial_guess=initial_guess, initial_F=initial_F, initial_afm=initial_afm,
      filename=filename,
      solver_data_package=solver_data_package
    )
    actions, monitors, convergers = cellular_actions(dt)

    cellular = generic_loop(
      name = "Cellular", 
      actions = actions,
      convergers = convergers,  
      monitors = monitors
    )

    cellular.run(
      dt, 
      max_its = max_its,
      min_its = min_its, 
      max_it_err_is_allowed = 7,
      print_final = True,
      print_current = 1,
      start_from_action_index = 0
    )

    return dt
  else:
    solvers.ctint.slave_run(solver_data_package=solver_data_package, printout=False)
