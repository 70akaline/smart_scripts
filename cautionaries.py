import numpy

def impose_real_valued_in_imtime_numpy(Q):
  #print "impose_real_valued_in_imtime_numpy"
  nw, ni, nj = numpy.shape(Q)
  for i in range(ni):
    for j in range(nj):  
      Q[:,i,j] += numpy.conjugate(Q[::-1,i,j])
  Q /= 2.0

def impose_real_valued_in_imtime(Q):
  impose_real_valued_in_imtime_numpy(Q.data)

def impose_equillibrium_numpy(Q):
  #print "impose_real_valued_in_imtime_numpy"
  for i in range(numpy.shape(Q)[0]):
    Q[i,:,:] += numpy.transpose(Q[i,:,:])
  Q /= 2.0

def impose_equilibrium(Q):
  impose_equillibrium_numpy(Q.data)

def impose_ph_symmetry_on_G_iw(Q):
  maxs = []
  for name,q in Q:
    maxs.append(numpy.amax(numpy.abs(q.data[:,:,:].real)))
    q.data[:,:,:] = 1j*q.data[:,:,:].imag
  print "impose_ph_symmetry_on_G_iw: max real part:", numpy.amax(numpy.array(maxs))

def symmetrize_cluster_nambu_Sigma(Sigma_IaJb_imp_iw, identical_pairs, su2=False, Us=None):
  print "symmetrize_cluster_nambu_Sigma"  
  if su2: assert not (Us is None), "in case of su2 symmetry, you must provide Us"
  err = False
  niws, nI, nI = numpy.shape(Sigma_IaJb_imp_iw.data)
  assert nI % 2 ==0, "in Nambu space nI must be even"
  nsites = nI/2
  for shift in ([0] if su2 else [0, nsites]):
    for ips in identical_pairs:
      total = numpy.zeros((niws),dtype=numpy.complex_)
      for i,j in ips:        
        total += Sigma_IaJb_imp_iw.data[:,i+shift,j+shift]
        if su2: total += -Us[i]*(i==j)-numpy.conj(Sigma_IaJb_imp_iw.data[:,i+nsites,j+nsites]) 
      total /=  len(ips)*(2 if su2 else 1)
      for i,j in ips:        
        err = numpy.any(numpy.greater(numpy.abs(Sigma_IaJb_imp_iw.data[:,i+shift,j+shift]-total[:]), 5e-3))
        if su2: err = err or numpy.any(numpy.greater(numpy.abs(-Us[i]*(i==j)-numpy.conj(Sigma_IaJb_imp_iw.data[:,i+nsites,j+nsites])-total[:]), 5e-3))
        if err: print "symmetrize_cluster_nambu_Sigma: WARNING!! i,j=%s,%s far from average"%(i,j)
        Sigma_IaJb_imp_iw.data[:,i+shift,j+shift] = total[:]
        if su2: Sigma_IaJb_imp_iw.data[:,i+nsites,j+nsites] = -Us[i]*(i==j)-numpy.conj(total[:])
  return err

def impose_su2_and_latt_inv_on_nambu_Sigma(Sigma_IaJb_imp_iw, Us):
  print "impose_su2_and_latt_inv_on_nambu_Sigma"
  err = False
  niws, nI, nI = numpy.shape(Sigma_IaJb_imp_iw.data)
  assert nI % 2 ==0, "in Nambu space nI must be even"
  nsites = nI/2
  for i in range(nsites):
    for j in range(i,nsites):
      total = numpy.zeros((niws),dtype=numpy.complex_)
      total += Sigma_IaJb_imp_iw.data[:,i,j] + Sigma_IaJb_imp_iw.data[:,j,i]
      total += -2.0*Us[i]*(i==j)-numpy.conj(Sigma_IaJb_imp_iw.data[:,i+nsites,j+nsites])-numpy.conj(Sigma_IaJb_imp_iw.data[:,j+nsites,i+nsites])  
      total /= 4.0 
      err = numpy.any(numpy.greater(numpy.abs(Sigma_IaJb_imp_iw.data[:,i,j]-total[:]), 5e-3))
      err = err or numpy.any(numpy.greater(numpy.abs(Sigma_IaJb_imp_iw.data[:,j,i]-total[:]), 5e-3))
      err = err or numpy.any(numpy.greater(numpy.abs(-Us[i]*(i==j)-numpy.conj(Sigma_IaJb_imp_iw.data[:,i+nsites,j+nsites])-total[:]), 5e-3))
      err = err or numpy.any(numpy.greater(numpy.abs(-Us[i]*(i==j)-numpy.conj(Sigma_IaJb_imp_iw.data[:,j+nsites,i+nsites])-total[:]), 5e-3))
      if err: print "impose_su2_and_latt_inv_on_nambu_Sigma: WARNING!! i,j=%s,%s far from average"%(i,j)
      Sigma_IaJb_imp_iw.data[:,i,j] = total[:]
      Sigma_IaJb_imp_iw.data[:,j,i] = total[:]
      Sigma_IaJb_imp_iw.data[:,i+nsites,j+nsites] = -Us[i]*(i==j)-numpy.conj(total[:])
      Sigma_IaJb_imp_iw.data[:,j+nsites,i+nsites] = -Us[i]*(i==j)-numpy.conj(total[:])
  return err

def impose_ph_symmetry_on_square_cluster_nambu_Sigma(Sigma_IaJb_imp_iw, Us):
  print "impose_ph_symmetry_on_square_cluster_nambu_Sigma"
  niws, nI, nI = numpy.shape(Sigma_IaJb_imp_iw.data)
  assert nI % 2 ==0, "in Nambu space nI must be even"
  nsites = nI/2
  L = int(numpy.sqrt(nsites))
  for i in range(nsites):
    for j in range(nsites):                
      rx = abs(i%L - j%L)
      ry = abs(i/L - j/L)
      if (rx+ry)%2==0: XX,unit = numpy.real,1 
      else: XX,unit = numpy.imag,1j  
      for shift in [0,nsites]:
        Sigma_IaJb_imp_iw.data[:,i+shift,j+shift] -= unit*XX(Sigma_IaJb_imp_iw.data[:,i+shift,j+shift]) + (i==j)*Us[i]/2.0

def impose_su2_and_inversion_symmetry_and_rotation_antisymmetry_on_anomalous_Sigma(Sigma_IaJb_imp_iw):
  print "impose_inversion_symmetry_and_rotation_antisymmetry_on_anomalous_Sigma"
  niws, nI, nI = numpy.shape(Sigma_IaJb_imp_iw.data)
  assert nI % 2 ==0, "in Nambu space nI must be even"
  nsites = nI/2
  assert nsites==4, "bigger clusters don't guarantee this property!"
  tot = numpy.zeros((niws),dtype=numpy.complex_)
  for i,j in [(0,1),(1,0),(2,3),(3,2)]:    
    tot+=Sigma_IaJb_imp_iw.data[:,i,j+nsites].real
    tot+=Sigma_IaJb_imp_iw.data[:,i+nsites,j].real
  for i,j in [(0,2),(2,0),(1,3),(3,1)]:
    tot-=Sigma_IaJb_imp_iw.data[:,i,j+nsites].real
    tot-=Sigma_IaJb_imp_iw.data[:,i+nsites,j].real
  tot/=16.0
  for i,j in [(0,1),(1,0),(2,3),(3,2)]:    
    Sigma_IaJb_imp_iw.data[:,i,j+nsites] = tot
    Sigma_IaJb_imp_iw.data[:,i+nsites,j] = tot
  for i,j in [(0,2),(2,0),(1,3),(3,1)]:
    Sigma_IaJb_imp_iw.data[:,i,j+nsites] = -tot
    Sigma_IaJb_imp_iw.data[:,i+nsites,j] = -tot

  for i,j in [(0,3),(3,0),(1,2),(2,1)]:
    Sigma_IaJb_imp_iw.data[:,i,j+nsites] = 0.0
    Sigma_IaJb_imp_iw.data[:,i+nsites,j] = 0.0
  for i in range(nsites):
    Sigma_IaJb_imp_iw.data[:,i,i+nsites] = 0.0
    Sigma_IaJb_imp_iw.data[:,i+nsites,i] = 0.0

