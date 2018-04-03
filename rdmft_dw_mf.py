from smart_scripts import *
import sys
sys.path.insert(0,'/home/jaksa/parallel_inverse/build')
from parallel_inverse import *

def initSquareNabmuCubicTBH(H0, Nx, Ny, eps, t, F, g, cyclic=True):
  nsites = Nx*Ny
  #H0 = numpy.zeros((2*nsites,2*nsites))
  H0[:nsites,:nsites] = initCubicTBH(Nx, Ny, 1, eps, t, cyclic)
  H0[nsites:,nsites:] = initCubicTBH(Nx, Ny, 1, -eps, -t, cyclic)
  H0[:nsites,nsites:] = g*F
  H0[nsites:,:nsites] = g*F

def get_nns(Nx,Ny,l): #find linear indices of the 4 nearest neighbors of site with linear index l ona cyclic square cluster Nx x Ny
  i = l % Nx
  j = int(l / Nx)
  lrn, lln, lbn, ltn = l+1,l-1,l+Nx,l-Nx
  if i+1==Nx: lrn = j*Nx
  if i==0: lln = (j+1)*Nx-1
  if j+1==Ny: lbn = i
  if j==0: ltn = (Ny-1)*Nx + i
  return [lrn,lln,lbn,ltn]

def create_nn_list(Nx,Ny):
  nsites = Nx*Ny  
  nn_list = numpy.zeros((nsites,4),dtype=numpy.int_)
  for l in range(nsites):
    nn_list[l,:] = get_nns(Nx,Ny,l)

def create_nn_bond_blocks(Nx,Ny):
  nsites = Nx*Ny  
  nn_bond_blocks = []
  for l in range(nsites):
    nns = get_nns(Nx,Ny,l)
    for nni in [0,2]:
      nn_bond_blocks.append("%.4d-%.4d"%(l,nns[nni]))
  return nn_bond_blocks

def initialize_F(Nx,Ny, forbidden_list=[], value=1.0):
  nsites = Nx*Ny 
  F = numpy.zeros((nsites,nsites))
  for l in range(nsites):
    if l in forbidden_list: continue
    lnns = get_nns(Nx,Ny,l)
    for lnni,lnn in enumerate(lnns):
      if lnn in forbidden_list: continue
      F[l,lnn] = (1 if (lnni<2) else -1)*value
  return F

def get_F(F, F_iw, F_tau, G_iajb_iw, Nx,Ny, forbidden_list=[]):
  #print "get_F:" 
  nsites = Nx*Ny 
  #print "nsites:", nsites  
  for nnbb, f_iw in F_iw:
    llnn = nnbb.split("-")
    l, lnn = int(llnn[0]), int(llnn[1])
    #print "l,lnn:",l,lnn
    if (l in forbidden_list) or (lnn in forbidden_list): continue
    f_iw << G_iajb_iw['up'][l+nsites,lnn]
    #fit_fermionic_gf_tail(f_iw, starting_iw=14.0, no_loc=True, overwrite_tail=False, max_order=5)    
    F_tau[nnbb] << InverseFourier(f_iw)
    #print "F_tau[nnbb].data[0,0,0].real",F_tau[nnbb].data[0,0,0].real 
    F[l,lnn] = F_tau[nnbb].data[0,0,0].real
    F[lnn,l] = F_tau[nnbb].data[0,0,0].real
      

def rdmft_dw_mf_data( niw, ntau, Nx,Ny, beta, blocks = ['up'] ):
  nsites = Nx*Ny
  dt = data() 
  dt.niw = niw
  dt.ntau = ntau
  dt.nsites = nsites
  dt.Nx = Nx
  dt.Ny = Ny
  dt.beta = beta
  dt.T = 1.0/beta
  dt.blocks = blocks 
  print "Adding lattice Green's functions..."  
  #AddGfData(dt, ['G_ij_iw', 'Sigma_ij_iw', 'G0_ij_iw'], blocks, nsites, niw, beta, domain = 'iw', suffix='', statistic='Fermion')
  #AddGfData(dt, ['G_ij_iw', 'G0_ij_iw'], blocks, nsites, niw, beta, domain = 'iw', suffix='', statistic='Fermion') #optimizing, we don't really need Sigma_ij_iw
  AddGfData(dt, ['G_iajb_iw'], blocks, 2*nsites, niw, beta, domain = 'iw', suffix='', statistic='Fermion') #optimizing further, we don't even need G0_ij_iw
  dt.iws = numpy.array([iw.value for iw in dt.G_iajb_iw[blocks[0]].mesh]) 
  dt.nn_bond_blocks = create_nn_bond_blocks(Nx,Ny)
  AddGfData(dt, ['F_iw'], dt.nn_bond_blocks, 1, niw, beta, domain = 'iw', suffix='', statistic='Fermion')
  AddGfData(dt, ['F_tau'], dt.nn_bond_blocks, 1, ntau, beta, domain = 'tau', suffix='', statistic='Fermion')
 
  print "Adding imp Green's functions..."  
  dt.combo_blocks = [ "%s|%.4d"%(block,i) for i in range(nsites) for block in blocks]
  AddGfData(dt, ['Sigma_imp_iw','Gweiss_iw'], dt.combo_blocks, 1, niw, beta, domain = 'iw', suffix='', statistic='Fermion')
  AddGfData(dt, ['Sigma_imp_tau','Gweiss_tau'], dt.combo_blocks, 1, ntau, beta, domain = 'tau', suffix='', statistic='Fermion')
  print "Adding imp numpy arrays..."  
  AddNumpyData(dt, ['H0'], (2*nsites,2*nsites))
  AddNumpyData(dt, ['F'], (nsites,nsites))
  AddBlockScalarData(dt, dt.combo_blocks, ['mus','Us'], vals=None)
  print "Done preparing containers"  
  return dt

def rdmft_dw_mf_set_params_and_initialize(dt, U, C, g, t=-0.25, Us_array=None, initial_guess='metal', initial_Sigma_imp_iw=None, initial_F=0.0, filename=None):
  dt.U = U
  dt.C = C
  dt.t = t
  dt.g = g
 
  if filename is None: 
    filename = "rdmft.%dx%d.U%.4f.T%.4f.C%.7f.from_%s_%s"\
                %(dt.Nx,dt.Ny,U,dt.T,C,initial_guess,('sc' if (initial_F!=0.0) else 'normal'))
  dt.archive_name = filename
  dt.dump = lambda dct: DumpData(dt, filename, Qs=[], exceptions=['G_iajb_iw'], dictionary=dct)
  dt.dump_final = lambda dct: DumpData(dt, filename, Qs=[], exceptions=['G_iajb_iw'], dictionary=dct)

  def get_block_from_combo_block(cb):
    return cb.split('|')[0]
  dt.get_block_from_combo_block = get_block_from_combo_block

  def get_site_index_from_combo_block(cb):   
    cbi = dt.combo_blocks.index(cb)
    return cbi % dt.nsites
  dt.get_site_index_from_combo_block = get_site_index_from_combo_block

  print "Filling zero in Sigma_imp_iw"
  dt.Sigma_imp_iw << 0.0

  nsites=dt.nsites
  Nx = dt.Nx
  Ny = dt.Ny
  dt.Us_array = {}
  for b in dt.blocks:
    if Us_array is None:
      print "Choosing Us..."    
      dt.Us_array[b] = numpy.array([U for i in range(nsites)])
      nni = int(C*nsites)
      print "nni:",nni
      print "nsites:",nsites
      print "actual C:", (1.0*nni)/(1.0*nsites)
      indices_to_be_made_zero = random.sample(range(nsites), nni)
      for i in indices_to_be_made_zero:
        dt.Us_array[b][i]=0.0
      print "dt.Us_array[b]:\n\n"
      for i in range(Ny):
        s=""
        for j in range(Nx): 
          if dt.Us_array[b][i*Nx+j]==0.0: s+='0'
          else: s+='X'
        print s
      print '\n'
    else:
      dt.Us_array[b] = Us_array[:]  
    #dt.Us_array[b] = numpy.random.choice([U,0], size=nsites, p=[C, 1-C]) 
  print "Making H0..."   
  dt.forbidden_list = [l for l,u in enumerate(dt.Us_array['up']) if u==0.0]
  print "forbidden_list:",dt.forbidden_list
  print "initial_F:", initial_F  
  dt.F[:,:] = initialize_F(Nx,Ny, forbidden_list=dt.forbidden_list, value=initial_F)
  #print "initialized F:",dt.F
  dt.get_H0()
  #print "initialized H0:",dt.H0
  if not (initial_Sigma_imp_iw is None):
    print "Filling in Sigma_imp_iw from initial_Sigma_imp_iw..."
    dt.Sigma_imp_iw << initial_Sigma_imp_iw

  if initial_guess=='metal':
    print "Making G..."
    dt.get_G_iajb_iw()
  #dt.Sigma_ij_iw << 0.0

  #dt.G_ij_iw << dt.G0_ij_iw

  for cb in dt.combo_blocks:
    block = get_block_from_combo_block(cb)
    site_index = get_site_index_from_combo_block(cb)
    dt.mus[cb] = 0.0
    dt.Us[cb] = dt.Us_array[block][site_index]
    if initial_Sigma_imp_iw is None:
      if initial_guess=='metal': dt.Gweiss_iw[cb] << dt.G_iajb_iw[block][site_index,site_index]
      elif initial_guess=='atomic': dt.Gweiss_iw[cb] << inverse(iOmega_n)  
      else: 
        try:
          LoadData(dt, initial_guess, 'current')
        except: 
          assert False, 'initial guess '+str(initial_guess)+' not implemented, or corrupted filename!'
      dt.get_Gweiss_tau(cb)
  print "Done initializing, about to dump..."
  dt.dump('initial')

def rdmft_dw_mf_set_calc(dt):
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

  #def memory_optimized_orbital_space_dyson():
  #  for b in dt.blocks:
  #    iws = numpy.array([iw.value for iw in dt.G_ij_iw[b].mesh]) 
  #    for iwi, iw in enumerate(iws):
  #      #dt.G_ij_iw[b].data[iwi,:,:] = numpy.linalg.inv(dt.G0_ij_iw[b].data[iwi,:,:])
  #      dt.G_ij_iw[b].data[iwi,:,:] = -dt.H0[b][:,:]
  #    for i in range(dt.nsites):
  #      dt.G_ij_iw[b].data[:,i,i] += iws - dt.Sigma_imp_iw["%s|%.4d"%(b,i)].data[:,0,0]
  #    #dt.G_ij_iw[b] << inverse(dt.G_ij_iw[b]) 
  #    for iwi, iw in enumerate(iws):
  #      dt.G_ij_iw[b].data[iwi,:,:] = numpy.linalg.inv(dt.G_ij_iw[b].data[iwi,:,:])
  #    impose_real_valued_in_imtime(dt.G_ij_iw[b])
  #    impose_equilibrium(dt.G_ij_iw[b])  
  #    fit_fermionic_gf_tail(dt.G_ij_iw[b], starting_iw=14.0, no_loc=False, overwrite_tail=False, max_order=5)

  dt.get_F = lambda: get_F(dt.F, dt.F_iw, dt.F_tau, dt.G_iajb_iw, dt.Nx, dt.Ny, forbidden_list=dt.forbidden_list)
  dt.get_H0 = lambda: initSquareNabmuCubicTBH(dt.H0, dt.Nx, dt.Ny, 0, dt.t, dt.F, dt.g, cyclic=True) 

  def parallel_optimized_orbital_space_dyson():
    for b in dt.blocks:
      iws = numpy.array([iw.value for iw in dt.G_iajb_iw[b].mesh]) 
      for iwi, iw in enumerate(iws):
        dt.G_iajb_iw[b].data[iwi,:,:] = -dt.H0[:,:]
      for i in range(dt.nsites):
        dt.G_iajb_iw[b].data[:,i,i] += iws - dt.Sigma_imp_iw["%s|%.4d"%(b,i)].data[:,0,0]
        dt.G_iajb_iw[b].data[:,i+dt.nsites,i+dt.nsites] += iws + numpy.conj(dt.Sigma_imp_iw["%s|%.4d"%(b,i)].data[:,0,0])
    invert(dt.G_iajb_iw)
    parallel_impose_real_valued_in_imtime(dt.G_iajb_iw)
    parallel_impose_equillibrium(dt.G_iajb_iw)

    #for b in dt.blocks:
      #impose_real_valued_in_imtime(dt.G_ij_iw[b])
    #  impose_equilibrium(dt.G_ij_iw[b])  
    #  fit_fermionic_gf_tail(dt.G_ij_iw[b], starting_iw=14.0, no_loc=False, overwrite_tail=False, max_order=5)
    #parallel_fit_tails( dt.G_ij_iw, 
    #                beta=dt.beta, 
    #                starting_iw=14.0, 
    #                no_hartree=False,
    #                no_loc=False,
    #                max_order = 5,
    #                overwrite_tail=False)

  #dt.get_G_ij_iw = lambda: orbital_space_dyson_get_G(dt.G_ij_iw, dt.G0_ij_iw, dt.Sigma_ij_iw)
  dt.get_G_iajb_iw = parallel_optimized_orbital_space_dyson
  #dt.get_G_iajb_iw = memory_optimized_orbital_space_dyson    

  dt.get_Gweiss_iw = lambda cb: orbital_space_dyson_get_G0(
    dt.Gweiss_iw[cb],
    dt.G_iajb_iw[dt.get_block_from_combo_block(cb)][dt.get_site_index_from_combo_block(cb),dt.get_site_index_from_combo_block(cb)],
    dt.Sigma_imp_iw[cb]
  )

  def get_Gweiss_tau(cb):
    impose_real_valued_in_imtime(dt.Gweiss_iw[cb])  
    fit_fermionic_gf_tail(dt.Gweiss_iw[cb], starting_iw=14.0, no_loc=False, overwrite_tail=True, max_order=5)
    dt.Gweiss_tau[cb] << InverseFourier(dt.Gweiss_iw[cb]) 
  dt.get_Gweiss_tau = get_Gweiss_tau

def rdmft_dw_mf_actions(dt):
  def impurity(dt):
    for cb in dt.combo_blocks:
      if dt.Us[cb]==0.0: continue
      dt.get_Sigma_imp_tau(cb)
      dt.get_Sigma_imp_iw(cb)

  def lattice(dt):
    #dt.get_Sigma_ij_iw()
    dt.get_G_iajb_iw()

  def post_lattice(dt):    
    dt.get_F()
    dt.get_H0()

  def pre_impurity(dt):
    for cb in dt.combo_blocks:
       dt.get_Gweiss_iw(cb) 
    impose_ph_symmetry_on_G_iw(dt.Gweiss_iw)
    for cb in dt.combo_blocks:
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
      printout = lambda data, it: (data.dump(it) if (int(it[-3:])%5==0) else 0),
      short_timings = True
    ),
    generic_action( 
      name = "post_lattice",
      main = post_lattice,
      mixers = [],#[lambda data, it: 0],
      cautionaries = [],#[lambda data, it: 0], allowed_errors = [],               
      printout = lambda data, it: 0,
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
      monitored_quantity = lambda: numpy.amax(numpy.abs(dt.F.real)), 
      h5key = 'Fmax_vs_it', 
      archive_name = dt.archive_name
    ),
    monitor(
      monitored_quantity = lambda: ( numpy.sum(numpy.abs(dt.F[numpy.nonzero(dt.F)].real) )/numpy.count_nonzero(dt.F) ), 
      h5key = 'Favg_vs_it', 
      archive_name = dt.archive_name
    )
  ]

  convergers = [
    #converger( monitored_quantity = lambda: dt.G_ij_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_G_ij_iw'),
    #converger( monitored_quantity = lambda: dt.Sigma_ij_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_ij_iw')
    converger( monitored_quantity = lambda: dt.Sigma_imp_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_Sigma_imp_iw'),
    converger( monitored_quantity = lambda: dt.F_iw, accuracy=1e-4, func=None, archive_name=dt.archive_name, h5key='diffs_F_iw')
  ]

  return actions, monitors, convergers

def rdmft_dw_mf_launcher(Nx, Ny, U, T, C, g,
                   Us_array = None, 
                   t=-0.25, 
                   initial_guess='metal',
                   initial_Sigma_imp_iw=None,
                   initial_F=1.0,
                   max_its = 10,
                   min_its = 5, 
                   iw_cutoff=30.0,
                   filename=None ):
  print "------------------- Welcome to RDMFT! -------------------------"
  beta = 1.0/T
  niw = int(((iw_cutoff*beta)/math.pi-1.0)/2.0)
  ntau = 3*niw
  print "Automatic niw:",niw
  
  nsites = Nx*Ny 
  dt = rdmft_dw_mf_data( niw, ntau, Nx, Ny, beta )
  rdmft_dw_mf_set_calc(dt)
  rdmft_dw_mf_set_params_and_initialize(dt, U, C, g, t, Us_array, initial_guess, initial_Sigma_imp_iw, initial_F, filename)
  actions, monitors, convergers = rdmft_dw_mf_actions(dt)

  rdmft_dw_mf = generic_loop(
    name = "RDMFT", 
    actions = actions,
    convergers = convergers,  
    monitors = monitors
  )

  rdmft_dw_mf.run(
    dt, 
    max_its = max_its,
    min_its = min_its, 
    max_it_err_is_allowed = 7,
    print_final = True,
    print_current = 1,
    start_from_action_index = (1 if (initial_Sigma_imp_iw is None) else 0)
  )

  return dt
