from pytriqs.archive import *
from pytriqs.gf import *
import numpy

#------------------ basic -------------------------------------------#

def initCubicTBH(Nx, Ny, Nz, eps, t, cyclic=True):
  H = [[0 for j in range(Nx*Ny*Nz)] for i in range(Nx*Ny*Nz)]  
  for i in range(Nx*Ny*Nz):
    H[i][i]=eps    
  for i in range(Nx):
    for j in range(Ny):
      for k in range(Nz): 
        if Nx>1:
          if i+1==Nx:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + j*Nx ] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + j*Nx + i+1 ] = t
        
          if i==0:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + j*Nx + Nx-1] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + j*Nx + i-1] = t  
            
        if Ny>1:
          if j+1==Ny:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + i ] = t
          else:  
            H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + (j+1)*Nx + i ] = t
        
          if j==0:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + (Ny-1)*Nx + i ] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + (j-1)*Nx + i ] = t

        if Nz>1:
          if (k+1==Nz): 
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ j*Nx + i ] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ (k+1)*Nx*Ny + j*Nx + i ] = t 
            
          if k==0:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ (Nz-1)*Nx*Ny + j*Nx + i ] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ (k-1)*Nx*Ny + j*Nx + i ] = t
    
  return H 

#----------------------------------------------------------------------------#

#------------------ various Dyson -------------------------------------------#

def orbital_space_dyson_get_G(G_ij_iw, G0_ij_iw, Sigma_ij_iw):
    G_ij_iw << inverse(inverse(G0_ij_iw)-Sigma_ij_iw)

def orbital_space_dyson_get_Sigma(Sigma_ij_iw, G0_ij_iw, G_ij_iw):
    Sigma_ij_iw << inverse(G0_ij_iw)-inverse(G_ij_iw)

def orbital_space_dyson_get_G0(G0_ij_iw, G_ij_iw, Sigma_ij_iw, mu_shift = 0 ):
    G0_ij_iw << inverse(mu_shift + inverse(G_ij_iw)+Sigma_ij_iw)

#------------------ IPT specific -------------------------------------------#

def get_Sigma_imp_tau_from_Gweiss_tau(Sigma_imp_tau, Gweiss_tau, U):
  assert numpy.shape(Sigma_imp_tau.data)==numpy.shape(Gweiss_tau.data), "get_Sigma_imp_tau_from_Gweiss_tau: Sigma and Gweiss unequal data structure"
   
  ntau = numpy.shape(Sigma_imp_tau.data)[0]
  Sigma_imp_tau.data[:,:,:] = U**2.0 * (Gweiss_tau.data[:,:,:])**2.0 * Gweiss_tau.data[::-1,:,:]

#---------------------mu search--------------------------------------------------------#

def search_for_mu(get_mu, set_mu, get_n, n, ph_symmetry, accepted_mu_range=[-3.0,3.0]):
  if mpi.is_master_node(): print "GW_mains: lattice:  n: ",n,", ph_symmetry",ph_symmetry, "accepted mu_range: ",accepted_mu_range

  if (n is None) or ((n==0.5) and ph_symmetry):
    if mpi.is_master_node(): print "no mu search to be performed! it is your duty to set the chemical potential to U/2. mu =",get_mu()
    print 'n on the lattice : ', get_n()
  else:
    def func(var):
      mu = var[0]        
      set_mu(mu)
      actual_n = get_n()
      val = 1.0-abs(n - actual_n)  
      if mpi.is_master_node(): print "amoeba func call: mu: %.2f desired n: %.2f actual n: %.2f val = "%(mu,n,actual_n),val
      if val != val: return -1e+6
      else: return val

    if mpi.is_master_node(): print "about to do mu search:"

    guesses = [get_mu(), 0.0, -0.1, -0.3, -0.4, -0.5, -0.7, 0.3, 0.5, 0.7]
    found = False  
    for l in range(len(guesses)):
      varbest, funcvalue, iterations = amoeba(var=[guesses[l]],
                                            scale=[0.01],
                                            func=func, 
                                            data = None,
                                            itmax=30,
                                            ftolerance=1e-2,
                                            xtolerance=1e-2,
                                            known_max = 1.0,
                                            known_max_accr = 5e-5)
      if (varbest[0]>accepted_mu_range[0] and varbest[0]<accepted_mu_range[1]) and (abs(funcvalue-1.0)<1e-2): #change the bounds for large doping
        found = True 
        func(varbest)
        break 
      if l+1 == len(guesses):
        if mpi.is_master_node(): print "mu search FAILED: doing a scan..."

        mu_grid = numpy.linspace(-1.0,0.3,50)
        func_values = [func(var=[mu]) for mu in mu_grid]
        if mpi.is_master_node(): 
          print "func_values: "
          for i in range(len(mu_grid)):
            print "mu: ",mu_grid[i], " 1-abs(n-n): ", func_values[i]
        mui_max = numpy.argmax(func_values)
        if mpi.is_master_node(): print "using mu: ", mu_grid[mui_max]
        set_mu(mu_grid[mui_max])  
        get_n()
           
    if mpi.is_master_node() and found:
      print "guesses tried: ", l  
      print "mu best: ", varbest
      print "1-abs(diff n - data.n): ", funcvalue
      print "iterations used: ", iterations

