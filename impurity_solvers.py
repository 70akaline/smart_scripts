#from functools import partial
import math, time, cmath
#from math import cos, exp, sin, log, log10, pi, sqrt
#import random
import numpy
#from numpy import matrix, array, zeros, identity
#from numpy.linalg import inv
from pytriqs.operators import *
from pytriqs.archive import *
#from pytriqs.gf.local import *
#from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi

from copy import deepcopy

#from first_include import *
#from tail_fitters import symmetrize_blockgf
#from tail_fitters import selective_symmetrize_blockgf
#from tail_fitters import selective_symmetrize_blockmatrix
#from tail_fitters import fit_and_overwrite_tails_on_Sigma
#from tail_fitters import fit_and_overwrite_tails_on_G

#from cthyb_spin import Solver

#from triqs_cthyb import *
try:
  from triqs_cthyb import Solver as CthybSolver
except:
  if mpi.is_master_node():
    print "CTHYB not installed"
  
try:
  from triqs_ctint import SolverCore as Solver
except:
  if mpi.is_master_node():
    print "CTINT not installed"

from slave_run import slave_run
from Kspace_plaquette import Kspace_plaquette

#try:
#  from pytriqs.applications.impurity_solvers.cthyb import SolverCore as cthybSolver
#  #from cthyb import SolverCore as cthybSolver
#except:
#  if mpi.is_master_node():
#    print "CTHYB not installed"

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
      beta = None,
      nsites = None,
      niw = None,
      ntau = 100000, 
    ):
      if solver_data_package is None: solver_data_package = {}    

      if nambu:
        gf_struct = {'nambu': range(2*nsites)}
      else:
        gf_struct = {'up': range(nsites), 'dn': range(nsites)}

      assert ntau>2*niw, "solvers.ctint.initialize_solvers: ERROR! ntau too small!!" 

      solver_data_package['constructor_parameters']={}
      solver_data_package['constructor_parameters']['beta'] = beta
      solver_data_package['constructor_parameters']['n_iw'] = niw
      solver_data_package['constructor_parameters']['n_tau'] = ntau
      solver_data_package['constructor_parameters']['gf_struct'] = gf_struct
      solver_data_package['tag'] = 'construct'

      if mpi.is_master_node(): print "solver_data_package:", solver_data_package  

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
      
      gf_struct = {}
      for bn in block_names:
        gf_struct[bn] = range(N_states)
     
      if nambu:
        nsites = N_states/2
        assert nsites==len(Us), " must be: nsites==len(Us)!!!"
        h_int = -Us[0] * n(block_names[0],0)*n(block_names[0],nsites)
        for i in range(1,nsites):
          h_int += -Us[i] * n(block_names[0],i)*n(block_names[0],i+nsites)
      else:
        assert N_states==len(Us), " must be: N_states==len(Us)!!!"
        h_int = Us[0] * n(block_names[0],0)*n(block_names[1],0)
        for i in range(1,N_states):
          h_int += Us[i] * n(block_names[0],i)*n(block_names[1],i)


      N_s = 2      
      if nambu:
        ALPHA = [
            [ [ -alpha + delta*(-1)**(s) for s in range(N_s)] for i in range(nsites) ]
          + [ [ -alpha - delta*(-1)**(s) for s in range(N_s)] for i in range(nsites) ]           
        ]
      else:
        ALPHA = [ [ [ alpha + delta*(-1)**(s+sig) for s in range(N_s)] for i in range(N_states)] for sig in range(2) ]
      if solver_data_package is None:  solver_data_package = {}    

      solver_data_package['solve_parameters'] = {}
      solver_data_package['solve_parameters']['Us'] = Us
      solver_data_package['solve_parameters']['alpha'] = ALPHA
      solver_data_package['solve_parameters']['n_cycles'] = n_cycles
      solver_data_package['solve_parameters']['max_time'] = max_time
      solver_data_package['solve_parameters']['length_cycle'] = 50
      solver_data_package['solve_parameters']['n_warmup_cycles'] = 2000
      solver_data_package['solve_parameters']['measure_M_tau'] = True
      solver_data_package['solve_parameters']['post_process'] = True
      solver_data_package['solve_parameters']['measure_histogram'] = True

      print solver_data_package['solve_parameters']
       
      solver_data_package['G0_iw'] = solver.G0_iw

      solver_data_package['tag'] = 'run'

      if mpi.size>1: 
         if mpi.is_master_node(): print "broadcasting solver_data_package!!"
         solver_data_package = mpi.bcast(solver_data_package)

      if mpi.is_master_node(): print "about to run "
      dct = deepcopy(solver_data_package['solve_parameters'])
      del dct['Us']
      try:
        solver.solve(h_int = h_int, **dct )
      except Exception as e:
        A = HDFArchive('black_box','w')
        A['solver']=solver
        del A
        raise e
      if mpi.is_master_node(): print "average sign: ",solver.average_sign


    @staticmethod  
    def slave_run(solver_data_package, printout=True, additional_tasks = {}):
      internal_data = {}
      def construct(solver_data_package):
        if printout: print "[Node ",mpi.rank,"] constructing solvers!!!"
        internal_data['solver'] = Solver( **(solver_data_package['constructor_parameters']) )
        internal_data['gf_struct'] = solver_data_package['constructor_parameters']['gf_struct']

      def run(solver_data_package):
        solver = internal_data['solver']
        gf_struct = internal_data['gf_struct']

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
          solver = internal_data['solver']
          gf_struct = internal_data['gf_struct']

          dct = deepcopy(solver_data_package['solve_parameters'])
          del dct['Us']
          if printout: print "[Node ",mpi.rank,"] about to run..."
          solver.solve(h_int = h_int, **dct )

          if printout: print "[Node ",mpi.rank,"] finished running successfully!"

        except Exception as e:
          print "[Node ",mpi.rank,"] ERROR: crash during running solver" 

      tasks = {
        'construct': construct,
        'run': run
      }
      tasks.update(additional_tasks)
      slave_run(solver_data_package, printout=False, tasks = tasks)

#    @staticmethod  
#    def slave_run(solver_data_package, printout=True, additional_tasks = {"tag": lambda: 0 }):
#      while True:
#        if printout: print "[Node ",mpi.rank,"] waiting for instructions..."
#
#        solver_data_package = mpi.bcast(solver_data_package)

#        if printout: print "[Node ",mpi.rank,"] received instructions!!!"

#        if solver_data_package is None: 
#          if printout: print "[Node ",mpi.rank,"] solver_data_package is None, will exit now. Goodbye."          
#          break

#        if solver_data_package['construct|run|exit'] == 0:     
#          if printout: print "[Node ",mpi.rank,"] constructing solvers!!!"
#          solver = Solver( **(solver_data_package['constructor_parameters']) )
#          gf_struct = solver_data_package['constructor_parameters']['gf_struct']

#        elif solver_data_package['construct|run|exit'] == 1:     
#          if printout: print "[Node ",mpi.rank,"] about to run..."
#          solver.G0_iw << solver_data_package['G0_iw']
#          Us = solver_data_package['solve_parameters']['Us']
#          block_names = gf_struct.keys()
#          if len(block_names)==1: nambu=True
#          else: nambu=False 
#          N_states = len(gf_struct[block_names[0]])
#          if nambu:
#            nsites = N_states/2
#            h_int = -Us[0] * n(block_names[0],0)*n(block_names[0],nsites)
#            for i in range(1,nsites):
#              h_int += -Us[i] * n(block_names[0],i)*n(block_names[0],i+nsites)
#          else:
#            h_int = Us[0] * n(block_names[0],0)*n(block_names[1],0)
#            for i in range(1,N_states):
#              h_int += Us[i] * n(block_names[0],i)*n(block_names[1],i)
     
#          try:
#            dct = deepcopy(solver_data_package['solve_parameters'])
#            del dct['Us']
#            solver.solve(h_int = h_int, **dct )

#            if printout: print "[Node ",mpi.rank,"] finished running successfully!"
#          except Exception as e:
#            print "[Node ",mpi.rank,"] ERROR: crash during running solver" 

#        elif solver_data_package['construct|run|exit'] == 2: 
#          if printout: print "[Node ",mpi.rank,"] received exit signal, will exit now. Goodbye."          
#          break

#        elif solver_data_package['construct|run|exit'] in additional_tasks.keys():
#          if printout: print "[Node ",mpi.rank,"] received additional task signal:",solver_data_package['construct|run|exit'] 
#          additional_tasks[solver_data_package['construct|run|exit']](solver_data_package)
      
#        else:
#          print "[Node ",mpi.rank,"] ERROR: unknown task tag!!!!" 
          

    @staticmethod
    def dump(solver, archive_name, suffix=''):    
      dct = {
        'mc_sign': solver.average_sign,
        'G_iw': solver.G_iw,
        'Sigma_iw': solver.Sigma_iw,
        'G0_iw': solver.G0_iw,
        'G0_shift_iw': solver.G0_shift_iw,
        'M_tau': solver.M_tau,
        'M_iw': solver.M_iw,
        'histogram': solver.histogram
      }     
      A = HDFArchive(archive_name)
      A['solver%s'%suffix] = dct










######################################################################################################################3

######################################################################################################################3

######################################################################################################################3

####### Kspace_nambu_cthyb:       
  class Kspace_nambu_cthyb:
    @staticmethod
    def initialize_solver(
      Q_IaJb_iw_template,
      solver_data_package = None,  
      ntau = 100000, 
    ):
      if solver_data_package is None: solver_data_package = {}    

      niw = len(Q_IaJb_iw_template.data[:,0,0])/2
      beta = Q_IaJb_iw_template.beta

      get_K_container, get_gf_struct, get_h_int, convert_to_K_space, convert_to_IJ_space = Kspace_plaquette(Q_IaJb_iw_template)

      gf_struct = get_gf_struct()

      assert ntau>2*niw, "solvers.ctint.initialize_solvers: ERROR! ntau too small!!" 

      solver_data_package['constructor_parameters']={}
      solver_data_package['constructor_parameters']['beta'] = beta
      solver_data_package['constructor_parameters']['n_iw'] = niw
      solver_data_package['constructor_parameters']['n_tau'] = ntau
      solver_data_package['constructor_parameters']['gf_struct'] = gf_struct
      solver_data_package['tag'] = 'construct'

      if mpi.is_master_node(): print "solver_data_package:", solver_data_package  

      if mpi.size>1: solver_data_package = mpi.bcast(solver_data_package)

      return CthybSolver( **solver_data_package['constructor_parameters'] )

    @staticmethod
    def run(
      solver, 
      U,    
      G0_IaJb_iw,
      n_cycles=20000,
      max_time = 5*60,
      solver_data_package = None,
      only_sign = False
    ):
     
      if solver_data_package is None:  solver_data_package = {}    

      solver_data_package['solve_parameters'] = {}
      solver_data_package['solve_parameters']['U'] = U
      solver_data_package['solve_parameters']['max_time'] = max_time
      solver_data_package['solve_parameters']["random_name"] = ""
      solver_data_package['solve_parameters']["length_cycle"] = 50
      solver_data_package['solve_parameters']["n_warmup_cycles"] = 50#0
      solver_data_package['solve_parameters']["n_cycles"] = 100000000
      solver_data_package['solve_parameters']["measure_G_l"] = True
      solver_data_package['solve_parameters']["move_double"] = True
      solver_data_package['solve_parameters']["perform_tail_fit"] = True
      solver_data_package['solve_parameters']["fit_max_moment"] = 2

      print solver_data_package['solve_parameters']
       
      solver_data_package['G0_IaJb_iw'] = G0_IaJb_iw

      solver_data_package['tag'] = 'run'

      if mpi.size>1: 
         if mpi.is_master_node(): print "broadcasting solver_data_package!!"
         solver_data_package = mpi.bcast(solver_data_package)

      if mpi.is_master_node(): print "about to run "
      dct = deepcopy(solver_data_package['solve_parameters'])
      del dct['U']

      get_K_container, get_gf_struct, get_h_int, convert_to_K_space, convert_to_IJ_space = Kspace_plaquette(G0_IaJb_iw)
      convert_to_K_space( solver.G0_iw, G0_IaJb_iw )
      h_int = get_h_int(U)
      try:
        solver.solve(h_int = h_int, **dct )
        Sigma_IaJb_iw = G0_IaJb_iw.copy()
        convert_to_IJ_space(Sigma_IaJb_iw, solver.Sigma_iw)
        return Sigma_IaJb_iw
      except Exception as e:
        A = HDFArchive('black_box','w')
        A['solver']=solver
        del A
        raise e
      if mpi.is_master_node(): print "average sign: ",solver.average_sign



    @staticmethod  
    def slave_run(solver_data_package, printout=True, additional_tasks = {}):
      internal_data = {}
      def construct(solver_data_package):
        if printout: print "[Node ",mpi.rank,"] constructing solvers!!!"
        internal_data['solver'] = CthybSolver( **(solver_data_package['constructor_parameters']) )
        internal_data['gf_struct'] = solver_data_package['constructor_parameters']['gf_struct']
      def run(solver_data_package):
        solver = internal_data['solver']
        U = solver_data_package['solve_parameters']['U']
        G0_IaJb_iw = solver_data_package['G0_IaJb_iw'].copy()

        get_K_container, get_gf_struct, get_h_int, convert_to_K_space, convert_to_IJ_space = Kspace_plaquette(G0_IaJb_iw)
        convert_to_K_space( solver.G0_iw, G0_IaJb_iw )
        h_int = get_h_int(U)
   
        try:
          dct = deepcopy(solver_data_package['solve_parameters'])
          del dct['U']
          if printout: print "[Node ",mpi.rank,"] about to run..."
          solver.solve(h_int = h_int, **dct )

          if printout: print "[Node ",mpi.rank,"] finished running successfully!"

        except Exception as e:
          print "[Node ",mpi.rank,"] ERROR: crash during running solver" 

      tasks = {
        'construct': construct,
        'run': run
      }
      tasks.update(additional_tasks)
      print "[ Node",mpi.rank,"]: tasks: ",tasks
      slave_run(solver_data_package, printout=False, tasks = tasks)
        
    @staticmethod
    def dump(solver, archive_name, suffix=''):    
      dct = {
        'G_iw': solver.G_iw,
        'G_tau': solver.G_tau,
        'Sigma_iw': solver.Sigma_iw,
        'G0_iw': solver.G0_iw,
        'G_l': solver.G_l
      }     
      A = HDFArchive(archive_name)
      A['solver%s'%suffix] = dct 





















######################################################################################################################3

######################################################################################################################3

######################################################################################################################3

####### cthyb:       
  class cthyb:
    @staticmethod
    def initialize_solver(
      nambu=False,
      solver_data_package = None,  
      beta = None,
      nsites = None,
      niw = None,
      ntau = 100000, 
    ):
      if solver_data_package is None: solver_data_package = {}    

      if nambu:
        gf_struct = [['nambu', range(2*nsites)]]
      else:
        gf_struct = [['up', range(nsites)], ['dn', range(nsites)]]

      assert ntau>2*niw, "solvers.ctint.initialize_solvers: ERROR! ntau too small!!" 

      solver_data_package['constructor_parameters']={}
      solver_data_package['constructor_parameters']['beta'] = beta
      solver_data_package['constructor_parameters']['n_iw'] = niw
      solver_data_package['constructor_parameters']['n_tau'] = ntau
      solver_data_package['constructor_parameters']['gf_struct'] = gf_struct
      solver_data_package['tag'] = 'construct'

      if mpi.is_master_node(): print "solver_data_package:", solver_data_package  

      if mpi.size>1: solver_data_package = mpi.bcast(solver_data_package)

      return CthybSolver( **solver_data_package['constructor_parameters'] )

    @staticmethod
    def run(
      solver, 
      Us,      
      nambu=False,
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
      
      gf_struct = {}
      for bn in block_names:
        gf_struct[bn] = range(N_states)
     
      if nambu:
        nsites = N_states/2
        assert nsites==len(Us), " must be: nsites==len(Us)!!!"
        h_int = -Us[0] * n(block_names[0],0)*n(block_names[0],nsites)
        for i in range(1,nsites):
          h_int += -Us[i] * n(block_names[0],i)*n(block_names[0],i+nsites)
      else:
        assert N_states==len(Us), " must be: N_states==len(Us)!!!"
        h_int = Us[0] * n(block_names[0],0)*n(block_names[1],0)
        for i in range(1,N_states):
          h_int += Us[i] * n(block_names[0],i)*n(block_names[1],i)

      if solver_data_package is None:  solver_data_package = {}    

      solver_data_package['solve_parameters'] = {}
      solver_data_package['solve_parameters']['Us'] = Us
      solver_data_package['solve_parameters']['max_time'] = max_time
      solver_data_package['solve_parameters']["random_name"] = ""
      solver_data_package['solve_parameters']["random_seed"] = 123 * mpi.rank + 567
      solver_data_package['solve_parameters']["length_cycle"] = 50
      solver_data_package['solve_parameters']["n_warmup_cycles"] = 50#0
      solver_data_package['solve_parameters']["n_cycles"] = 10000000
      solver_data_package['solve_parameters']["measure_g_l"] = True
      solver_data_package['solve_parameters']["move_double"] = True
      solver_data_package['solve_parameters']["perform_tail_fit"] = True
      solver_data_package['solve_parameters']["fit_max_moment"] = 2

      print solver_data_package['solve_parameters']
       
      solver_data_package['G0_iw'] = solver.G0_iw

      solver_data_package['tag'] = 'run'

      if mpi.size>1: 
         if mpi.is_master_node(): print "broadcasting solver_data_package!!"
         solver_data_package = mpi.bcast(solver_data_package)

      if mpi.is_master_node(): print "about to run "
      dct = deepcopy(solver_data_package['solve_parameters'])
      del dct['Us']
      try:
        solver.solve(h_int = h_int, **dct )
      except Exception as e:
        A = HDFArchive('black_box','w')
        A['solver']=solver
        del A
        raise e
      if mpi.is_master_node(): print "average sign: ",solver.average_sign


    @staticmethod  
    def slave_run(solver_data_package, printout=True, additional_tasks = {}):
      internal_data = {}
      def construct(solver_data_package):
        if printout: print "[Node ",mpi.rank,"] constructing solvers!!!"
        internal_data['solver'] = CthybSolver( **(solver_data_package['constructor_parameters']) )
        internal_data['gf_struct'] = solver_data_package['constructor_parameters']['gf_struct']
      def run(solver_data_package):
        solver = internal_data['solver']
        gf_struct = internal_data['gf_struct']

        solver.G0_iw << solver_data_package['G0_iw']
        Us = solver_data_package['solve_parameters']['Us']
        block_names = [ bl[0] for bl in gf_struct ]
        print "block names: ",block_names
        if len(block_names)==1: nambu=True
        else: nambu=False 
        N_states = len(gf_struct[0][1])
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
          solver = internal_data['solver']
          gf_struct = internal_data['gf_struct']

          dct = deepcopy(solver_data_package['solve_parameters'])
          del dct['Us']
          if printout: print "[Node ",mpi.rank,"] about to run..."
          solver.solve(h_int = h_int, **dct )

          if printout: print "[Node ",mpi.rank,"] finished running successfully!"

        except Exception as e:
          print "[Node ",mpi.rank,"] ERROR: crash during running solver" 


      tasks = {
        'construct': construct,
        'run': run
      }
      tasks.update(additional_tasks)
      print "[ Node",mpi.rank,"]: tasks: ",tasks
      slave_run(solver_data_package, printout=False, tasks = tasks)
        
    @staticmethod
    def dump(solver, archive_name, suffix=''):    
      dct = {
        'G_iw': solver.G_iw,
        'G_tau': solver.G_tau,
        'Sigma_iw': solver.Sigma_iw,
        'G0_iw': solver.G0_iw,
        'G_l': solver.G_l
      }     
      A = HDFArchive(archive_name)
      A['solver%s'%suffix] = dct 
