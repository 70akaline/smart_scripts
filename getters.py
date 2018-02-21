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

#------------------ various Dyson -------------------------------------------#

def orbital_space_dyson_get_G(G_ij_iw, G0_ij_iw, Sigma_ij_iw):
    G_ij_iw << inverse(inverse(G0_ij_iw)-Sigma_ij_iw)

def orbital_space_dyson_get_Sigma(Sigma_ij_iw, G0_ij_iw, G_ij_iw):
    Sigma_ij_iw << inverse(G0_ij_iw)-inverse(G_ij_iw)

def orbital_space_dyson_get_G0(G0_ij_iw, G_ij_iw, Sigma_ij_iw ):
    G0_ij_iw << inverse(inverse(G_ij_iw)+Sigma_ij_iw)

#------------------ IPT specific -------------------------------------------#

def get_Sigma_imp_tau_from_Gweiss_tau(Sigma_imp_tau, Gweiss_tau, U):
  assert numpy.shape(Sigma_imp_tau.data)==numpy.shape(Gweiss_tau.data), "get_Sigma_imp_tau_from_Gweiss_tau: Sigma and Gweiss unequal data structure"
   
  ntau = numpy.shape(Sigma_imp_tau.data)[0]
  for taui in range(ntau):
     Sigma_imp_tau.data[taui,:,:] = U**2.0 * (Gweiss_tau.data[taui,:,:])**2.0 * Gweiss_tau.data[-1-taui,:,:]
