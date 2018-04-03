from functools import partial
import math, time, cmath
from math import cos, exp, sin, log, log10, pi, sqrt
import random
import numpy
from numpy import matrix, array, zeros, identity
from numpy.linalg import inv
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi

from copy import deepcopy

from first_include import *
from tail_fitters import symmetrize_blockgf
from tail_fitters import selective_symmetrize_blockgf
from tail_fitters import selective_symmetrize_blockmatrix
from tail_fitters import fit_and_overwrite_tails_on_Sigma
from tail_fitters import fit_and_overwrite_tails_on_G

#from cthyb_spin import Solver

try:
  from triqs_ctint import SolverCore as Solver
except:
  if mpi.is_master_node():
    print "CTINT not installed"

try:
  from pytriqs.applications.impurity_solvers.cthyb import SolverCore as cthybSolver
  #from cthyb import SolverCore as cthybSolver
except:
  if mpi.is_master_node():
    print "CTHYB not installed"

#from selfconsistency.useful_functions import adjust_n_points
#from selfconsistency.provenance import hash_dict

import copy

################################ IMPURITY #########################################

class solvers:
  class ctint:
    @staticmethod
    def initialize_solver(
      nambu=False,
      solver_data_package = None,  
      nsites = None,
      niw = None,
      ntau = 2000, 
    ):
      if solver_data_package is None: solver_data_package = {}    

      if nambu:
        gf_struct = {'nambu': range(2*nsites)}
      else:
        gf_struct = {'up': range(nsites), 'dn': range(nsites)}

      assert ntau>2*niw, "solvers.ctint.initialize_solvers: ERROR! ntau too small!!" 

      solver_data_package['constructor_parameters']={}
      solver_data_package['constructor_parameters']['beta'] = data.beta
      solver_data_package['constructor_parameters']['n_iw'] = niw
      solver_data_package['constructor_parameters']['n_tau'] = ntau
      solver_data_package['constructor_parameters']['gf_struct'] = gf_struct
      solver_data_package['construct|run|exit'] = 0

      if mpi.size>1: solver_data_package = mpi.bcast(solver_data_package)

      return Solver( **solver_data_package['constructor_parameters'] )

    @staticmethod
    def run(
      solver, 
      Us,      
      nambu=False,
      alpha=0.5,
      delta=0.1,
      n_cycles=20000,
      max_time = 5*60,
      solver_data_package = None,
      only_sign = False
    ):

      block_names = [name for name,g in solver.G0_iw]
      if nambu: assert len(block_names)==1, "in Nambu we have one block!!"
      else: assert len(block_names)==2, "we need two blocks!!"
      N_states = len(solver.G0_iw[block_names[0]].data[0,0,:])
      if nambu: assert N_states % 2 == 0, "in nambu there needs to be an even number of states" 
      assert len(N_states)==len(Us), " must be: len(N_states)==len(Us)!!!"
      gf_struct = {}
      for bn in block_names:
        gf_struct[bn] = range(N_states)
     
      if nambu:
        nsites = N_states/2
        h_int = -Us[0] * n(block_names[0],0)*n(block_names[0],nsites)
        for i in range(1,nsites):
          h_int += -Us[i] * n(block_names[0],i)*n(block_names[0],i+nsites)
      else:
        h_int = Us[0] * n(block_names[0],0)*n(block_names[1],0)
        for i in range(1,N_states):
          h_int += Us[i] * n(block_names[0],i)*n(block_names[1],i)


      N_s = 2      
      if nambu:
        ALPHA = [ [ [ alpha + delta*(-1)**(s+sig) for s in range(N_s)] for i in range(N_states)] for sig in range(2) ]
      else:
        ALPHA = [
            [ [ alpha + delta*(-1)**(s) for s in range(N_s)] for i in range(nsites) ]
          + [ [ alpha - delta*(-1)**(s) for s in range(N_s)] for i in range(nsites) ]           
        ]

      if solver_data_package is None:  solver_data_package = {}    

      solver_data_package['solve_parameters'] = {}
      solver_data_package['solve_parameters']['Us'] = Us
      solver_data_package['solve_parameters']['alpha'] = ALPHA
      solver_data_package['solve_parameters']['n_cycles'] = n_cycles
      solver_data_package['solve_parameters']['max_time'] = max_time
      solver_data_package['solve_parameters']['length_cycle'] = 200
      solver_data_package['solve_parameters']['n_warmup_cycles'] = 2000
      solver_data_package['solve_parameters']['measure_M_tau'] = True
      solver_data_package['solve_parameters']['post_process'] = True

      print solver_data_package['solve_parameters']
       
      solver_data_package['G0_iw'] = solver.G0_iw

      solver_data_package['construct|run|exit'] = 1

      if mpi.size>1: 
         if mpi.is_master_node(): print "broadcasting solver_data_package!!"
         solver_data_package = mpi.bcast(solver_data_package)

      if mpi.is_master_node(): print "about to run "
      dct = deepcopy(solver_data_package['solve_parameters'])
      del dct['Us']
      solver.solve(h_int = h_int, **dct )
      if mpi.is_master_node(): print "average sign: ",solver.average_sign

    @staticmethod
    def slave_run(solver_data_package, printout=True):
      while True:
        if printout: print "[Node ",mpi.rank,"] waiting for instructions..."

        solver_data_package = mpi.bcast(solver_data_package)

        if printout: print "[Node ",mpi.rank,"] received instructions!!!"

        if solver_data_package is None: 
          if printout: print "[Node ",mpi.rank,"] solver_data_package is None, will exit now. Goodbye."          
          break

        if solver_data_package['construct|run|exit'] == 0:     
          if printout: print "[Node ",mpi.rank,"] constructing solvers!!!"
          solver = Solver( **(solver_data_package['constructor_parameters']) )
          gf_struct = solver_data_package['constructor_parameters']['gf_struct']

        if solver_data_package['construct|run|exit'] == 1:     
          if printout: print "[Node ",mpi.rank,"] about to run..."
          solver.G0_iw << solver_data_package['G0_iw']
          Us = solver_data_package['solve_parameters']['Us']
          block_names = gf_struct.keys()
          if len(block_names)==1: nambu=True
          else: nambu=False 
          N_states = len(gf_struct[block_names[0]])
          if nambu:
            nsites = N_states/2
            h_int = -Us[0] * n(block_names[0],0)*n(block_names[0],nsites)
            for i in range(1,nsites):
              h_int += -Us[i] * n(block_names[0],i)*n(block_names[0],i+nsites)
          else:
            h_int = Us[0] * n(block_names[0],0)*n(block_names[1],0)
            for i in range(1,N_states):
              h_int += Us[i] * n(block_names[0],i)*n(block_names[1],i)
     
          try:
            dct = deepcopy(solver_data_package['solve_parameters'])
            del dct['Us']
            solver.solve(h_int = h_int, **dct )

            if printout: print "[Node ",mpi.rank,"] finished running successfully!"
          except Exception as e:
            print "[Node ",mpi.rank,"] ERROR: crash during running solver" 

        if solver_data_package['construct|run|exit'] == 2: 
          if printout: print "[Node ",mpi.rank,"] received exit signal, will exit now. Goodbye."          
          break

    @staticmethod
    def dump(solver, archive_name, suffix=''):    
      dct = {
        'mc_sign': solver.average_sign
        'G_iw': solver.G_iw
        'Sigma_iw': solver.Sigma_iw
        'G0_iw': solver.G0_iw
        'M_tau': solver.M_tau
        'M_iw': solver.M_iw  
      }
      A = HDFArchive(archive_name)
      A['solver%s'%suffix] = dct
        
 
