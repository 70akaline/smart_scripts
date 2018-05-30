import pytriqs.utility.mpi as mpi
import socket
from smart_scripts import *
from copy import deepcopy
from parallel_dyson import parallel_get_Nambu_G_for_cellular
from parallel_dyson import optimized_parallel_get_Nambu_G_for_cellular

use_cthyb=False
use_Kspace_nambu_cthyb=False

def matrix_dispersion(Nx, Ny, t, kx, ky):
  A = initCubicTBH(Nx, Ny, 1, 0, t, cyclic=False)
  nsites = Nx*Ny
  B = numpy.zeros((nsites,nsites), dtype=numpy.complex_)
  for l in range(nsites):
    i = l % Nx
    j = int(l / Nx)
    if i+1==Nx: 
      lnn = j*Nx
      B[l,lnn]+=t*numpy.exp(1j*kx)
      B[lnn,l]+=t*numpy.exp(-1j*kx)
    if j+1==Ny: 
      lnn = i
      B[l,lnn]+=t*numpy.exp(1j*ky)
      B[lnn,l]+=t*numpy.exp(-1j*ky) 
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

def fill_H0_k(H0_k,Nx,Ny,t,ks, mus, Us, F=None):
  H0_k[:,:,:,:] = 0.0
  nsites = Nx*Ny 
  for kxi,kx in enumerate(ks):
    for kyi,ky in enumerate(ks):
      epsk = matrix_dispersion(Nx,Ny,t,kx,ky)
      H0_k[kxi,kyi,:nsites,:nsites] = epsk - numpy.diag(mus) + numpy.diag(Us)
      H0_k[kxi,kyi,nsites:,nsites:] = numpy.diag(mus) - epsk#numpy.conj(epsk)
      if not(F is None):
        H0_k[kxi,kyi,:nsites,nsites:] = F
        H0_k[kxi,kyi,nsites:,:nsites] = F


def get_G(G_IaJb_k_iw, H0_k, Sigma_IaJb_imp_iw):
  iws = [iw.value for iw in Sigma_IaJb_imp_iw.mesh]
  dummy1, nk, nk, nIa, nIa = numpy.shape(G_IaJb_k_iw)
  for iwi, iw in enumerate(iws):
    for kxi in range(nk):
      for kyi in range(nk):
        G_IaJb_k_iw[iwi,kxi,kyi,:,:] = numpy.linalg.inv(iw*numpy.eye(nIa)-H0_k[kxi,kyi,:,:]-Sigma_IaJb_imp_iw.data[iwi,:,:])

def mpi_parallel_get_G(solver_data_package, dt):  
  if mpi.is_master_node():
    print 'master node about to broadcast get_G_parameters...'
    solver_data_package['tag'] = 'get_G'
    solver_data_package['get_G_parameters'] = {}
    solver_data_package['get_G_parameters']['Sigma_IaJb_imp_iw'] = dt.Sigma_IaJb_imp_iw
    solver_data_package['get_G_parameters']['H0_k'] = dt.H0_k
    print "master node sending solver_data_package: ",solver_data_package.keys()
    solver_data_package = mpi.bcast(solver_data_package)

  if not (dt.master_rank is None): 
    print "[ master_rank",dt.master_rank,"]: received solver_data_package: ",solver_data_package.keys() 
    print "[ master_rank",dt.master_rank,"]: about to do get_G" 
    for iwii, iwi in enumerate(dt.iwis_per_master):
      dt.Sigma_imp_data[iwii,:,:] = solver_data_package['get_G_parameters']['Sigma_IaJb_imp_iw'].data[iwi,:,:]
    H0_k = solver_data_package['get_G_parameters']['H0_k']
    parallel_get_Nambu_G_for_cellular(numpy.array(dt.iws_per_master), H0_k, dt.Sigma_imp_data, dt.G_IaJb_k_iw)
    print "[ master_rank",dt.master_rank,"]: done doing get_G" 
  if mpi.is_master_node():
    del solver_data_package['get_G_parameters']

def mpi_parallel_get_G_loc(solver_data_package, dt):
  if mpi.is_master_node():
    print '[ master node ] [ master_rank',dt.master_rank ,'] about to broadcast get_G_loc_parameters...'
    dt.G_IaJb_loc_iw << 0.0
    solver_data_package['tag'] = 'get_G_loc'
    solver_data_package['get_G_loc_parameters'] = {}
    solver_data_package['get_G_loc_parameters']['G_IaJb_loc_iw'] = dt.G_IaJb_loc_iw
    solver_data_package = mpi.bcast(solver_data_package) 
  G_IaJb_loc_iw = solver_data_package['get_G_loc_parameters']['G_IaJb_loc_iw']
  if not (dt.master_rank is None): 
    print "[ master_rank",dt.master_rank,"]: about to do get_G_loc" 
    half_niw, nk, nk, nIa, nJb = numpy.shape(dt.G_IaJb_k_iw)
    G_IaJb_loc_iw.data[dt.iwis_per_master[0]:dt.iwis_per_master[-1]+1,:,:] = numpy.sum(dt.G_IaJb_k_iw, axis=(1,2))/nk**2 
  #print "[rank ", mpi.rank,"[ master_rank",dt.master_rank,"]: about to reduce G_IaJb_loc_iw" 
  G_IaJb_loc_iw = mpi.all_reduce(None, G_IaJb_loc_iw, None)
  if not (dt.master_rank is None): 
    print "[ master_rank",dt.master_rank,"]: done doing get_G_loc" 
  if mpi.is_master_node():
    dt.G_IaJb_loc_iw << G_IaJb_loc_iw  
    fit_fermionic_gf_tail(dt.G_IaJb_loc_iw, starting_iw=14.0, no_loc=False, overwrite_tail=True, max_order=5)
    dt.G_IaJb_loc_tau << InverseFourier(dt.G_IaJb_loc_iw)
    del solver_data_package['get_G_loc_parameters']
    print '[ master node ] [ master_rank',dt.master_rank ,'] done doing get_G_loc'

def optimized_mpi_parallel_get_G_loc(solver_data_package, dt):
  if mpi.is_master_node():
    print '[ master node ] [ master_rank',dt.master_rank ,'] about to broadcast get_G_loc_parameters...'
    dt.G_IaJb_loc_iw << 0.0
    solver_data_package['tag'] = 'optimized_get_G_loc'
    solver_data_package['optimized_get_G_loc_parameters'] = {}
    solver_data_package['optimized_get_G_loc_parameters']['G_IaJb_loc_iw'] = dt.G_IaJb_loc_iw
    solver_data_package['optimized_get_G_loc_parameters']['Sigma_IaJb_imp_iw'] = dt.Sigma_IaJb_imp_iw
    solver_data_package['optimized_get_G_loc_parameters']['H0_k'] = dt.H0_k
    solver_data_package = mpi.bcast(solver_data_package) 
  G_IaJb_loc_iw = solver_data_package['optimized_get_G_loc_parameters']['G_IaJb_loc_iw']
  Sigma_IaJb_imp_iw = solver_data_package['optimized_get_G_loc_parameters']['Sigma_IaJb_imp_iw']
  H0_k = solver_data_package['optimized_get_G_loc_parameters']['H0_k']
  if not (dt.master_rank is None): 
    print "[ master_rank",dt.master_rank,"]: about to do get_G_loc" 
    optimized_parallel_get_Nambu_G_for_cellular(
      dt.iws_per_master,
      H0_k,
      Sigma_IaJb_imp_iw.data[dt.iwis_per_master[0]:dt.iwis_per_master[-1]+1,:,:],
      G_IaJb_loc_iw.data[dt.iwis_per_master[0]:dt.iwis_per_master[-1]+1,:,:]
    )
  #print "[rank ", mpi.rank,"[ master_rank",dt.master_rank,"]: about to reduce G_IaJb_loc_iw" 
  G_IaJb_loc_iw = mpi.all_reduce(None, G_IaJb_loc_iw, None)
  if not (dt.master_rank is None): 
    print "[ master_rank",dt.master_rank,"]: done doing get_G_loc" 
  if mpi.is_master_node():
    dt.G_IaJb_loc_iw << G_IaJb_loc_iw  
    fit_fermionic_gf_tail(dt.G_IaJb_loc_iw, starting_iw=14.0, no_loc=False, overwrite_tail=True, max_order=5)
    dt.G_IaJb_loc_tau << InverseFourier(dt.G_IaJb_loc_iw)
    del solver_data_package['optimized_get_G_loc_parameters']
    print '[ master node ] [ master_rank',dt.master_rank ,'] done doing optimized_get_G_loc'

def get_G_loc(G_IaJb_loc_tau, G_IaJb_loc_iw, G_IaJb_k_iw):
  niws, nk, nk, nIa, nJb = numpy.shape(G_IaJb_k_iw)  
  G_IaJb_loc_iw.data[:,:,:] = numpy.sum(G_IaJb_k_iw, axis=(1,2))/nk**2 
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
      ns['up'].append(-G_IaJb_loc_tau.data[-1,Ia,Ia])
      ns['dn'].append(-G_IaJb_loc_tau.data[0,Ia+nsites,Ia+nsites])
  return ns

def get_Gweiss(Gweiss_IaJb_iw, G_IaJb_loc_iw, Sigma_IaJb_imp_iw):
  Gweiss_IaJb_iw << inverse(inverse(G_IaJb_loc_iw)+Sigma_IaJb_imp_iw)
  impose_real_valued_in_imtime(Gweiss_IaJb_iw)
  fit_fermionic_gf_tail(Gweiss_IaJb_iw, starting_iw=14.0, no_loc=False, overwrite_tail=True, max_order=5)

def get_Sigma(solver, Sigma_imp_iw, Gweiss_iw, Us, max_time=5*60, delta=0.1, solver_data_package=None, nambu=True):
  assert nambu, "not nambu not implemented"
  if not use_Kspace_nambu_cthyb:
    solver.G0_iw['nambu'] << Gweiss_iw 
  if use_cthyb:
    solvers.cthyb.run(
      solver=solver, 
      Us=Us,      
      nambu=nambu,
      n_cycles=100000000,
      max_time = max_time,
      solver_data_package = solver_data_package,
      only_sign = False
    )
  elif use_Kspace_nambu_cthyb:
    assert numpy.unique(Us).size==1, "Kspace_cthyb only with translational invariance within the supercell"
    Sigma_imp_iw << solvers.Kspace_nambu_cthyb.run(
      solver=solver, 
      U=Us[0],    
      G0_IaJb_iw=Gweiss_iw,
      n_cycles=100000000,
      max_time = max_time,
      solver_data_package = solver_data_package,
      only_sign = False
    )
  else:
    solvers.ctint.run(
      solver=solver, 
      Us=Us,      
      nambu=nambu,
      alpha=0.5,
      delta=delta,
      n_cycles=100000000,
      max_time = max_time,
      solver_data_package = solver_data_package,
      only_sign = False
    )
  if not use_Kspace_nambu_cthyb:
    Sigma_imp_iw << solver.Sigma_iw['nambu']

def cellular_data( nk, niw, ntau, Nx,Ny, beta, skip_lattice=False ):
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
  if not skip_lattice: 
    print "Adding lattice Green's function..."  
    AddNumpyData(dt, ['G_IaJb_k_iw'], (2*niw, nk,nk, 2*nsites, 2*nsites) ) #optimizing further, we don't even need G0_ij_iw
  else: print "Skipping lattice Green's function..."
  print "Adding loc and imp Green's functions..."  
  for Q in ['G_IaJb_loc_iw','Sigma_IaJb_imp_iw','Gweiss_IaJb_iw']:
    vars(dt)[Q] = GfImFreq(indices = range(2*nsites), beta = beta, n_points = niw, statistic = 'Fermion')
  for Q in ['G_IaJb_loc_tau']:
    vars(dt)[Q] = GfImTime(indices = range(2*nsites), beta = beta, n_points = ntau, statistic = 'Fermion')

  dt.iws = numpy.array([iw.value for iw in dt.G_IaJb_loc_iw.mesh]) 
  print "Adding Hamiltonian and parameters..."  
  AddNumpyData(dt, ['H0_k'], (nk,nk, 2*nsites,2*nsites))
  AddNumpyData(dt, ['mus','Us'], (nsites))
  print "Done preparing containers"  
  return dt

def get_master_rank():
  hostname = socket.gethostname()
  a = str(mpi.rank)+"|"+hostname+"|"  
  a = mpi.all_reduce(None, a, None)
  #if mpi.is_master_node(): print a
  rhs = a.split("|")
  #if mpi.is_master_node(): print rhs
  ranks = [rhs[2*i] for i in range(len(rhs)/2) ]
  hosts = [rhs[2*i+1] for i in range(len(rhs)/2) ]
  n_masters = len(list(set(hosts)))
  this_host_ranks = [r for ri,r in enumerate(ranks) if hosts[ri]==hostname]   
  if min([int(r) for r in this_host_ranks])==mpi.rank:
    #print "rank: ",mpi.rank," is the master of node:",hostname
    master_rank = list(set(hosts)).index(hostname)
    return master_rank, n_masters, hostname
  else: return None, n_masters, hostname

def mpi_parallel_add_lattice_data(dt, iws, nk, nsites):
  master_rank, n_masters, hostname = get_master_rank()
  dt.master_rank = master_rank
  if not (master_rank is None):
    niws = len(iws)
    assert niws % n_masters == 0, "number of freqs must be divisible by number of masters!!!" 
    niws_per_master = niws/n_masters 
    dt.iws_per_master = iws[ master_rank*niws_per_master : (master_rank+1)*niws_per_master ]
    dt.iwis_per_master = range( master_rank*niws_per_master,(master_rank+1)*niws_per_master )
    print "[ master_rank",master_rank,"]: dt.iws_per_master from",dt.iws_per_master[0],"to",dt.iws_per_master[-1]
    AddNumpyData(dt, ['G_IaJb_k_iw'], (niws_per_master, nk,nk, 2*nsites, 2*nsites) ) 
    #AddNumpyData(dt, ['H0_k'], (nk,nk, 2*nsites,2*nsites))
    AddNumpyData(dt, ['Sigma_imp_data'], (niws_per_master, 2*nsites,2*nsites))

def optimized_mpi_parallel_add_iw_ranges(dt, iws):
  master_rank, n_masters, hostname = get_master_rank()
  dt.master_rank = master_rank
  if not (master_rank is None):
    niws = len(iws)
    assert niws % n_masters == 0, "number of freqs must be divisible by number of masters!!!" 
    niws_per_master = niws/n_masters 
    dt.iws_per_master = iws[ master_rank*niws_per_master : (master_rank+1)*niws_per_master ]
    dt.iwis_per_master = range( master_rank*niws_per_master,(master_rank+1)*niws_per_master )
    print "[ master_rank",master_rank,"]: dt.iws_per_master from",dt.iws_per_master[0],"to",dt.iws_per_master[-1]

def cellular_set_calc( dt, max_time=5*60, delta=0.1, solver_data_package=None, nambu=True, su2=True ):
  #dt.get_H0_k = lambda: fill_H0_k(dt.H0_k,dt.Nx,dt.Ny,dt.t,dt.ks, dt.mus, dt.Us)
  dt.get_H0_k = lambda: fill_H0_k(dt.H0_k,dt.Nx,dt.Ny,dt.t,dt.ks, dt.mus, dt.Us, F=initialize_F(dt.Nx,dt.Ny, value=dt.initial_F))
  dt.get_Gweiss = lambda: get_Gweiss(dt.Gweiss_IaJb_iw, dt.G_IaJb_loc_iw, dt.Sigma_IaJb_imp_iw)

  dt.get_Sigma = lambda: get_Sigma(dt.solver, dt.Sigma_IaJb_imp_iw, dt.Gweiss_IaJb_iw, dt.Us, max_time, delta, solver_data_package, nambu)

  #dt.get_G = lambda: get_G(dt.G_IaJb_k_iw, dt.H0_k, dt.Sigma_IaJb_imp_iw)
  dt.get_G = lambda: parallel_get_Nambu_G_for_cellular(numpy.array(dt.iws), dt.H0_k, dt.Sigma_IaJb_imp_iw.data, dt.G_IaJb_k_iw)

  dt.get_G_loc = lambda: get_G_loc(dt.G_IaJb_loc_tau, dt.G_IaJb_loc_iw, dt.G_IaJb_k_iw)

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
  dt.mus_initial = deepcopy(mus)
  dt.initial_mus = numpy.array(mus)
  dt.initial_F = initial_F
  dt.fixed_n = fixed_n
  dt.n=n 
  dt.ph_symmetry = ph_symmetry
  dt.iteration = 0

  if filename is None: 
    filename = "cellular.%dx%d.U%.4f.T%.4f.C%.4f.from_%s_%s_%s"\
                %(dt.Nx,dt.Ny,U,dt.T,dt.C,initial_guess,('sc' if (initial_F!=0.0) else 'normal'),('afm' if (initial_afm!=0.0) else 'pm'))
  dt.archive_name = filename
  dt.dump = lambda dct: DumpData(dt, filename, Qs=[], exceptions=['solver','G_IaJb_k_iw'], dictionary=dct)
  dt.dump_final = lambda dct: DumpData(dt, filename, Qs=[], exceptions=['solver'], dictionary=dct)

  if use_cthyb:
    dt.solver = solvers.cthyb.initialize_solver(
      nambu=True,
      solver_data_package = solver_data_package,  
      beta = dt.beta,
      nsites = dt.nsites,
      niw = dt.niw,
      ntau = max(dt.ntau, 100001)
    )
  elif use_Kspace_nambu_cthyb:
    dt.solver = solvers.Kspace_nambu_cthyb.initialize_solver(
      Q_IaJb_iw_template = dt.Gweiss_IaJb_iw,
      solver_data_package = solver_data_package,  
      ntau = max(dt.ntau, 100001) 
    )
  else:
    dt.solver = solvers.ctint.initialize_solver(
      nambu=True,
      solver_data_package = solver_data_package,  
      beta = dt.beta,
      nsites = dt.nsites,
      niw = dt.niw,
      ntau = max(dt.ntau, 100001)
    )

  print "Making H0.."
#  if initial_F!=0.0:
#    fill_H0_k(dt.H0_k,dt.Nx,dt.Ny,dt.t,dt.ks, dt.mus, dt.Us, F=initialize_F(dt.Nx,dt.Ny, value=initial_F))
#  else: dt.get_H0_k()
  dt.get_H0_k()
  print "Filling Sigma_IaJb_imp_iw.."
  dt.Sigma_IaJb_imp_iw << 0.0
  for I,U in enumerate(Us):
    print "I:",I
    dt.Sigma_IaJb_imp_iw[I,I] << -U*0.5-initial_afm-int(initial_guess=='atomic')*inverse(iOmega_n)
    dt.Sigma_IaJb_imp_iw[I+dt.nsites,I+dt.nsites] << -U*0.5-initial_afm-int(initial_guess=='atomic')*inverse(iOmega_n)
  
  print "Getting G, Gloc and n. n on the lattice: ",dt.get_n()
  #dt.get_G()
  #dt.get_G_loc()
  print "Getting Gweiss.."
  dt.get_Gweiss() 
  #if initial_F!=0.0: dt.get_H0_k() #we don't need anymore the anomalous part in H0. It's only for the first iteration.
    
  print "Done initializing, about to dump..."
  dt.dump('initial')

def cellular_actions(dt, accr, iterations_to_keep_initial_F=0):
  def impurity(dt):
    dt.get_Sigma()

  def lattice(dt):
    if dt.iteration == iterations_to_keep_initial_F:
      print "---- iteration %d reached! setting dt.initial_F to zero..."%dt.iteration
      dt.initial_F = 0.0
      dt.get_H0_k()

    dt.iteration += 1

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
      printout = lambda data, it: ( 
        solvers.cthyb.dump(data.solver, dt.archive_name, suffix=it) 
        if (use_cthyb or use_Kspace_nambu_cthyb) else 
        solvers.ctint.dump(data.solver, dt.archive_name, suffix=it)
      ),
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
    converger( monitored_quantity = lambda: dt.Sigma_IaJb_imp_iw, accuracy=accr, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_IaJb_imp_iw'),
    converger( monitored_quantity = lambda: dt.G_IaJb_loc_iw, accuracy=accr, func=None, archive_name=dt.archive_name, h5key='diffs_G_IaJb_loc_iw'),
  ]

  return actions, monitors, convergers

def cellular_launcher(
     Nx, Ny,
     Us, T,
     mus=None,
     n=0.5, fixed_n=True,
     t=-0.25, 
     initial_guess='metal',
     initial_F=1.0, iterations_to_keep_initial_F=0,
     initial_afm=0.0,
     nk=24,
     n_cycles=100000,            
     max_time_rules= [ [1, 5*60], [2, 20*60], [4, 80*60], [8, 200*60], [16,400*60] ], time_rules_automatic=False, exponent = 0.7, overall_prefactor=1.0, no_timing = False,
     max_its = 10,
     min_its = 5, 
     accr = 1e-3,  
     iw_cutoff=30.0,
     filename=None,
     parallel_mem_for_lattice=False,
     use_optimized_parallel_mem=False
):
  solver_data_package={}
  nsites = Nx*Ny

  if mpi.is_master_node():
    print "------------------- Welcome to CELLULAR! -------------------------"
    beta = 1.0/T
    niw = int(((iw_cutoff*beta)/numpy.pi-1.0)/2.0)
    if mpi.size % 16 == 0:
      n_masters = mpi.size / 16          
      niw -= niw % (n_masters/2)
    ntau = 3*niw
    print "Automatic niw:",niw   

    Umax = numpy.max(Us) 
    if no_timing:
      max_time=-1
      print "no timing!!!"
    else:
      if time_rules_automatic:
        pref = ((beta/8.0)*Umax*nsites)**exponent #**1.2
        print "pref: ",pref 
        max_time = int(overall_prefactor*pref*5*60)
        print "max times automatic: ",max_time
      else:
        for r in max_time_rules:
          if r[0]<=nsites:
            max_time = r[1]
        print "max_time from rules: ",max_time

    if mus is None:
      mus = numpy.array(Us)/2.0 

    dt = cellular_data( nk, niw, ntau, Nx,Ny, beta, skip_lattice= parallel_mem_for_lattice )
    if parallel_mem_for_lattice:
      solver_data_package['iws'] = dt.iws.copy()
      if mpi.size>1: 
         solver_data_package = mpi.bcast(solver_data_package)
      if use_optimized_parallel_mem:
        optimized_mpi_parallel_add_iw_ranges(dt, dt.iws)
      else:
        mpi_parallel_add_lattice_data(dt, dt.iws, nk, nsites)  
      del solver_data_package['iws'] 

    cellular_set_calc( dt, max_time, delta=(0.0 if n==0.5 else 0.0), solver_data_package=solver_data_package, nambu=True )
    if parallel_mem_for_lattice:
      if use_optimized_parallel_mem:
        dt.get_G = lambda: None
        dt.get_G_loc = lambda: optimized_mpi_parallel_get_G_loc(solver_data_package, dt)
      else:
        dt.get_G = lambda: mpi_parallel_get_G(solver_data_package, dt)        
        dt.get_G_loc = lambda: mpi_parallel_get_G_loc(solver_data_package, dt)
    import sys
    sys.path.append('/home/jaksa/TRIQS/source/nested_scripts/')
    from nested_structure import get_identical_pair_sets
    dt.get_Sigma = lambda get_sig=dt.get_Sigma: [
      get_sig(), 
      ( symmetrize_cluster_nambu_Sigma(dt.Sigma_IaJb_imp_iw, get_identical_pair_sets(Nx,Ny), su2=True, Us=Us)
        if ((numpy.unique(Us).size==1)and(initial_afm==0.0)) else
        ( impose_su2_and_latt_inv_on_nambu_Sigma(dt.Sigma_IaJb_imp_iw, Us)
          if (initial_afm==0.0) else
          None
        )
      ),
      ( impose_ph_symmetry_on_square_cluster_nambu_Sigma(dt.Sigma_IaJb_imp_iw, Us)
        if ((n==0.5) and fixed_n and (initial_afm==0.0)) else
        None
      ),
      impose_real_valued_in_imtime(dt.Sigma_IaJb_imp_iw),
      ( impose_su2_and_inversion_symmetry_and_rotation_antisymmetry_on_anomalous_Sigma(dt.Sigma_IaJb_imp_iw)
        if ((numpy.unique(Us).size==1)and(initial_afm==0.0)) else
        ( impose_su2_and_cluster_symmetry_and_rotation_antisymmetry_on_anomalous_Sigma(dt.Sigma_IaJb_imp_iw) 
          if ((Us[3]==0.0) and (numpy.unique(Us[:-1]).size==1) and (len(Us)==4) and (initial_afm==0.0)) else
          None
        )
      )
    ]

    cellular_set_params_and_initialize(
      dt, Us=Us, mus=mus,
      n=n, fixed_n=fixed_n, ph_symmetry=True,
      t=t,
      initial_guess=initial_guess, initial_F=initial_F, initial_afm=initial_afm,
      filename=filename,
      solver_data_package=solver_data_package
    )
    actions, monitors, convergers = cellular_actions(dt, accr, iterations_to_keep_initial_F)

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
    if mpi.size>1:
      solver_data_package['tag'] = 'exit'
      solver_data_package = mpi.bcast(solver_data_package)
    return dt
  else:
    if use_cthyb: solver_class = solvers.cthyb
    elif use_Kspace_nambu_cthyb: solver_class = solvers.Kspace_nambu_cthyb
    else: solver_class = solvers.ctint 

    if parallel_mem_for_lattice:
      dt=data()
      solver_data_package = mpi.bcast(solver_data_package)
      if use_optimized_parallel_mem:
        optimized_mpi_parallel_add_iw_ranges(dt, solver_data_package['iws'])
      else:
        mpi_parallel_add_lattice_data(dt, solver_data_package['iws'], nk, nsites)  

      solver_class.slave_run(
        solver_data_package=solver_data_package,
        printout=False,
        additional_tasks = (
          { 'get_G': lambda sdp: mpi_parallel_get_G(sdp, dt),
            'get_G_loc': lambda sdp: mpi_parallel_get_G_loc(sdp, dt),
          } if not use_optimized_parallel_mem else
          { 'optimized_get_G_loc': lambda sdp: optimized_mpi_parallel_get_G_loc(sdp, dt)
          }
        )
      )
    else:
      solver_class.slave_run(solver_data_package=solver_data_package, printout=False)
