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
