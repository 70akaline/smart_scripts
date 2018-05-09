import math, time, cmath
from math import cos, exp, sin, log, log10, pi, sqrt
import random
import numpy
from numpy import matrix, array, zeros, identity
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi

import copy

def fit_fermionic_sigma_tail(Q, starting_iw=14.0, no_hartree=False, no_loc=False, max_order=5, overwrite_tail=True):
  Nc = len(Q.data[0,:,0])    
  if no_loc:
    known_coeff = TailGf(Nc,Nc,3,-1)
    known_coeff[-1] = numpy.zeros((Nc,Nc),dtype=numpy.complex)
    known_coeff[0] = numpy.zeros((Nc,Nc),dtype=numpy.complex)
    known_coeff[1] = numpy.zeros((Nc,Nc),dtype=numpy.complex)
  elif no_hartree:
    known_coeff = TailGf(Nc,Nc,2,-1)
    known_coeff[-1] = numpy.zeros((Nc,Nc))
    known_coeff[0] = numpy.zeros((Nc,Nc))
  else:
    known_coeff = TailGf(Nc,Nc,1,-1)
    known_coeff[-1] = numpy.zeros((Nc,Nc),dtype=numpy.complex)
  nmax = Q.mesh.last_index()
  nmin = int(((starting_iw*Q.beta)/math.pi-1.0)/2.0) 
  before = Q.data[-1,0,0]
  Q.fit_tail(known_coeff,max_order,nmin,nmax, overwrite_tail)  
  if (Q.data[-1,0,0]==before) and overwrite_tail:
    print "fit_fermionic_sigma_tail: WARNING: negative part of the tail may not have been overwritten!!!"

def fit_fermionic_gf_tail(Q, starting_iw=14.0, no_loc=False, overwrite_tail=False, max_order=5):
  Nc = len(Q.data[0,0,:])
  if no_loc:
    known_coeff = TailGf(Nc,Nc,2,-1)
    known_coeff[-1] = zeros((Nc,Nc))#array([[0.]])
    known_coeff[0] = zeros((Nc,Nc))#array([[0.]])
  else:
    known_coeff = TailGf(Nc,Nc,3,-1)
    known_coeff[-1] = zeros((Nc,Nc))#array([[0.]])
    known_coeff[0] = zeros((Nc,Nc))#array([[0.]])
    known_coeff[1] = identity(Nc)
  nmax = Q.mesh.last_index()
  nmin = int(((starting_iw*Q.beta)/math.pi-1.0)/2.0) 
  nmin = max(nmin,1)
  Q.fit_tail(known_coeff,max_order,nmin,nmax, overwrite_tail)

def fit_and_remove_constant_tail(Q, starting_iw=14.0, max_order = 5, general=False):
  Nc = len(Q.data[0,0,:])
  known_coeff = TailGf(Nc,Nc,1,-1)
  known_coeff[-1] = zeros((Nc,Nc))
  nmax = Q.mesh.last_index()
  nmin = int(((starting_iw*Q.beta)/math.pi-1.0)/2.0) 
  print "nmin, nmax: ", nmin, nmax
  nmin = max(1,nmin)
  Q.fit_tail(known_coeff,max_order,nmin,nmax)
  if general:
    for l1 in range(Nc):
      for l2 in range(Nc):
        tail0 = Q.tail[0][l1,l2]  
        Q[l1,l2] -= tail0
  else:
    for l in range(Nc):
      tail0 = Q.tail[0][l,l]  
      Q[l,l] -= tail0
  Q.fit_tail(known_coeff,max_order,nmin,nmax)
  return tail0


def fit_bosonic_tail(Q, starting_iw=14.0, no_static=False, overwrite_tail=False, max_order=5):
  Nc = len(Q.data[0,0,:])
  if no_static:
    known_coeff = TailGf(Nc,Nc,2,-1)
    known_coeff[-1] = zeros((Nc,Nc))#array([[0.]])
    known_coeff[0] = zeros((Nc,Nc))#array([[0.]])
  else:
    known_coeff = TailGf(Nc,Nc,1,-1)
    known_coeff[-1] = zeros((Nc,Nc))#array([[0.]])
  nmax = Q.mesh.last_index()
  nmin = int(((starting_iw*Q.beta)/math.pi-1.0)/2.0) 
  Q.fit_tail(known_coeff,max_order,nmin,nmax, overwrite_tail)

def prepare_G0_iw(G0_iw, Gweiss, fermionic_struct, starting_iw=14.0):
  known_coeff = TailGf(1,1,3,-1)
  known_coeff[-1] = array([[0.]])
  known_coeff[0] = array([[0.]])
  known_coeff[1] = array([[1.]])
  for U in fermionic_struct.keys():    
     G0_iw[U] << Gweiss[U]
     nmax = G0_iw[U].mesh.last_index()
     nmin = int(((starting_iw*G0_iw.beta)/math.pi-1.0)/2.0) 
     G0_iw[U].fit_tail(known_coeff,5,nmin,nmax, True)

def prepare_G0_iw_atomic(G0_iw, mus, fermionic_struct):
  known_coeff = TailGf(1,1,3,-1)
  known_coeff[-1] = array([[0.]])
  known_coeff[0] = array([[0.]])
  known_coeff[1] = array([[1.]])
  for U in fermionic_struct.keys():    
     G0_iw[U] << inverse(iOmega_n+mus[U])
     nmax = G0_iw[U].mesh.last_index()
     nmin = nmax/2
     G0_iw[U].fit_tail(known_coeff,5,nmin,nmax, True)

def extract_Sigma_from_F_and_G(Sigma_iw, F_iw, G_iw):
  Sigma_iw << inverse(G_iw)*F_iw

def extract_Sigma_from_G0_and_G(Sigma_iw, G0_iw, G_iw):
  Sigma_iw << inverse(G0_iw) - inverse(G_iw)

def fit_and_overwrite_tails_on_Sigma(Sigma_iw, starting_iw=14.0, no_hartree=False):
  Nc = len(Sigma_iw.data[0,0,:])
  if no_hartree:  
    known_coeff = TailGf(Nc,Nc,2,-1)
    known_coeff[-1] = zeros((Nc,Nc))#array([[0.]])
    known_coeff[0] = zeros((Nc,Nc))#array([[0.]])
  else:  
    known_coeff = TailGf(Nc,Nc,1,-1)
    known_coeff[-1] = zeros((Nc,Nc))#array([[0.]])
  nmax = Sigma_iw.mesh.last_index()
  nmin = int(((starting_iw*Sigma_iw.beta)/math.pi-1.0)/2.0) 
  Sigma_iw.fit_tail(known_coeff,5,nmin,nmax, True)

def fit_and_overwrite_tails_on_G(G_iw, starting_iw=14.0):
  Nc = len(G_iw.data[0,0,:])
  fixed_coeff = TailGf(Nc,Nc,3,-1)
  fixed_coeff[-1] = zeros((Nc,Nc))
  fixed_coeff[0] = zeros((Nc,Nc))
  fixed_coeff[1] = identity(Nc)
  nmax = G_iw.mesh.last_index()
  nmin = int(((starting_iw*G_iw.beta)/math.pi-1.0)/2.0) #the matsubara index at iw_n = starting_iw
  G_iw.fit_tail(fixed_coeff, 5, nmin, nmax, True)

def symmetrize_blockgf(Q):
  iterator = [(name,g) for name,g in Q]
  print "iterator:", iterator  
  block_names = [name for name,g in Q]
  print "block_names:", block_names
  Qcopy = copy.deepcopy(Q[block_names[0]])
  Qcopy << 0.0
  for key in block_names:
    print "key:",key
    print "Q[key]:",Q[key]
    Qcopy << Qcopy + Q[key]
  Qcopy /= len(block_names)
  for key in block_names:
    Q[key] << Qcopy
  del Qcopy

def selective_symmetrize_blockgf(Q, blocks):
  Qcopy = Q[blocks[0]].copy()
  Qcopy << 0.0
  for key in blocks:
    Qcopy += Q[key]
  Qcopy /= len(blocks)
  for key in blocks:
    Q[key] << Qcopy
  del Qcopy

def selective_symmetrize_blockmatrix(Q, blocks):
  Qcopy = Q[blocks[0]].copy()
  Qcopy[:,:] = 0.0
  for key in blocks:
    Qcopy += Q[key]
  Qcopy /= len(blocks)
  for key in blocks:
    Q[key] = Qcopy
  del Qcopy
